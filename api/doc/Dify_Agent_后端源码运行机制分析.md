# Dify Agent 后端源码运行机制分析

## 概述

本文档详细分析了 Dify Agent 的后端源码运行机制，以 GraphAgent-Hugegraph 为例，展示了从用户请求到工具调用的完整流程。

## 1. 系统架构概览

Dify Agent 采用分层架构设计：

```
用户请求 → Controller → Service → Generator → Runner → Tool Engine
```

### 核心组件

1. **Controller Layer**: 处理 HTTP 请求
2. **Service Layer**: 业务逻辑处理
3. **Generator Layer**: 应用生成器
4. **Runner Layer**: Agent 运行器
5. **Tool Engine**: 工具执行引擎

## 2. 配置文件分析

### GraphAgent-Hugegraph.yml 配置结构

```yaml
app:
  mode: agent-chat  # Agent 聊天模式
  name: GraphAgent-Hugegraph

model_config:
  agent_mode:
    enabled: true
    max_iteration: 10  # 最大迭代次数
    strategy: react    # ReAct 策略
    tools:             # 工具配置
    - tool_name: hugegraph_query
      tool_parameters:
        operation: get_schema
        limit: 10
    - tool_name: hugegraph_gremlin
      tool_parameters:
        gremlin_query: ''
        limit: ''

  model:
    name: gpt-5-chat-latest
    provider: langgenius/openai/openai
    mode: chat

  pre_prompt: |
    你是一个图专家，负责处理用户关于图谱的问题以及执行图查询任务。
    请严格按照以下步骤完成任务：
    1. 阅读用户问题 {{user_question}}，理解其查询意图。
    2. 如果用户提供了额外的上下文信息 {{user_context}}，结合上下文来优化查询。
    3. 根据意图构造合适的 Gremlin 查询语句，用于在 HugeGraph 数据库中执行。
    4. 使用工具 hugegraph_query 调用构造的查询。
    5. 在输出部分，先给出清晰的查询语句，再根据查询结果生成简洁的自然语言回答。

  user_input_form:
  - variable: user_question
    label: user_question
    type: text-input
  - variable: user_context  
    label: user_context
    type: text-input
  - variable: answer_style
    label: answer_style
    type: text-input
```

## 3. 请求处理流程

### 3.1 Controller 层 (completion.py)

```python
class ChatMessageApi(Resource):
    @get_app_model(mode=[AppMode.CHAT, AppMode.AGENT_CHAT])
    def post(self, app_model):
        # 1. 解析请求参数
        parser = reqparse.RequestParser()
        parser.add_argument("inputs", type=dict, required=True)
        parser.add_argument("query", type=str, required=True)
        parser.add_argument("model_config", type=dict, required=True)
        parser.add_argument("conversation_id", type=uuid_value)
        args = parser.parse_args()

        # 2. 日志记录请求参数
        logging.info("=== GraphAgent Chat Request ===")
        logging.info("App ID: %s", app_model.id)
        logging.info("Request Args: %s", args)

        # 3. 调用 AppGenerateService
        response = AppGenerateService.generate(
            app_model=app_model, 
            user=account, 
            args=args, 
            invoke_from=InvokeFrom.DEBUGGER, 
            streaming=streaming
        )
```

### 3.2 Service 层 (app_generate_service.py)

```python
class AppGenerateService:
    @classmethod
    def generate(cls, app_model: App, user: Union[Account, EndUser], 
                args: Mapping[str, Any], invoke_from: InvokeFrom, 
                streaming: bool = True):
        # 1. 系统级别限流检查
        if dify_config.BILLING_ENABLED:
            # 检查是否为免费计划
            limit_info = BillingService.get_info(app_model.tenant_id)
            if limit_info["subscription"]["plan"] == "sandbox":
                if cls.system_rate_limiter.is_rate_limited(app_model.tenant_id):
                    raise InvokeRateLimitError("Rate limit exceeded")

        # 2. 应用级别限流
        max_active_request = AppGenerateService._get_max_active_requests(app_model)
        rate_limit = RateLimit(app_model.id, max_active_request)
        
        # 3. 根据应用模式选择生成器
        if app_model.mode == AppMode.AGENT_CHAT.value or app_model.is_agent:
            return rate_limit.generate(
                AgentChatAppGenerator.convert_to_event_stream(
                    AgentChatAppGenerator().generate(
                        app_model=app_model, user=user, args=args, 
                        invoke_from=invoke_from, streaming=streaming
                    ),
                ),
                request_id,
            )
```

### 3.3 Generator 层 (agent_chat/app_generator.py)

```python
class AgentChatAppGenerator(MessageBasedAppGenerator):
    def generate(self, *, app_model: App, user: Union[Account, EndUser], 
                args: Mapping[str, Any], invoke_from: InvokeFrom, 
                streaming: bool = True):
        # 1. 参数验证
        if not streaming:
            raise ValueError("Agent Chat App does not support blocking mode")
        
        if not args.get("query"):
            raise ValueError("query is required")

        # 2. 获取或创建对话
        conversation = None
        conversation_id = args.get("conversation_id")
        if conversation_id:
            conversation = ConversationService.get_conversation(
                app_model=app_model, conversation_id=conversation_id, user=user
            )

        # 3. 获取应用模型配置
        app_model_config = self._get_app_model_config(
            app_model=app_model, conversation=conversation
        )

        # 4. 创建应用生成实体
        application_generate_entity = AgentChatAppGenerateEntity(
            task_id=str(uuid.uuid4()),
            app_config=app_config,
            model_conf=model_config,
            inputs=inputs,
            query=query,
            files=files,
            user_id=user.id,
            stream=streaming,
            invoke_from=invoke_from,
            # ... 其他参数
        )

        # 5. 创建队列管理器和运行器
        queue_manager = MessageBasedAppQueueManager(
            task_id=application_generate_entity.task_id,
            user_id=application_generate_entity.user_id,
            invoke_from=application_generate_entity.invoke_from,
            conversation_id=conversation.id,
            app_mode=AppMode.AGENT_CHAT,
            message_id=message.id,
        )

        # 6. 运行 Agent
        runner = AgentChatAppRunner()
        runner.run(
            application_generate_entity=application_generate_entity,
            queue_manager=queue_manager,
            conversation=conversation,
            message=message,
        )
```

## 4. Agent Runner 执行机制

### 4.1 AgentChatAppRunner (agent_chat/app_runner.py)

```python
class AgentChatAppRunner(AppRunner):
    def run(self, application_generate_entity: AgentChatAppGenerateEntity,
            queue_manager: AppQueueManager, conversation: Conversation,
            message: Message) -> None:
        
        # 1. 获取应用配置
        app_config = cast(AgentChatAppConfig, application_generate_entity.app_config)
        
        # 2. 组织提示消息
        prompt_messages, _ = self.organize_prompt_messages(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=dict(inputs),
            files=list(files),
            query=query,
            memory=memory,
        )

        # 3. 内容审核
        try:
            self.moderation_for_inputs(
                app_record.id,
                tenant_id=app_config.tenant_id,
                app_generate_entity=application_generate_entity,
                inputs=application_generate_entity.inputs,
                query=application_generate_entity.query,
            )
        except ModerationError as e:
            # 处理审核错误
            pass

        # 4. 创建 Agent 实体
        agent_entity = app_config.agent
        if not agent_entity:
            raise ValueError("Agent not found")

        # 5. 选择 Agent Runner 类型
        if agent_entity.strategy == AgentEntity.Strategy.REACT:
            runner_cls = CotChatAgentRunner
        elif agent_entity.strategy == AgentEntity.Strategy.FUNCTION_CALL:
            runner_cls = FunctionCallAgentRunner
        else:
            raise ValueError(f"Unknown agent strategy: {agent_entity.strategy}")

        # 6. 记录日志
        logging.info("Agent Max Iteration: %s", agent_entity.max_iteration)
        logging.info("Agent Tools: %s", [tool.tool_name for tool in agent_entity.tools])
        logging.info("Model Provider: %s", model_instance.provider)
        logging.info("Model Name: %s", model_instance.model)

        # 7. 创建并运行具体的 Runner
        runner = runner_cls(
            tenant_id=app_config.tenant_id,
            application_generate_entity=application_generate_entity,
            conversation=conversation_result,
            app_config=app_config,
            model_config=application_generate_entity.model_conf,
            config=agent_entity,
            queue_manager=queue_manager,
            message=message_result,
            user_id=application_generate_entity.user_id,
            memory=memory,
            prompt_messages=prompt_message,
            model_instance=model_instance,
        )

        # 8. 执行运行器
        invoke_result = runner.run(
            message=message,
            query=query,
            inputs=inputs,
        )

        # 9. 处理调用结果
        self._handle_invoke_result(
            invoke_result=invoke_result,
            queue_manager=queue_manager,
            stream=application_generate_entity.stream,
            agent=True,
        )
```

### 4.2 FunctionCallAgentRunner (fc_agent_runner.py)

这是核心的 Agent 执行器，实现了 Function Calling 模式：

```python
class FunctionCallAgentRunner(BaseAgentRunner):
    def run(self, message: Message, query: str, **kwargs: Any) -> Generator[LLMResultChunk, None, None]:
        # 1. 初始化工具实例
        tool_instances, prompt_messages_tools = self._init_prompt_tools()
        
        # 2. 设置迭代参数
        iteration_step = 1
        max_iteration_steps = min(app_config.agent.max_iteration, 99) + 1
        function_call_state = True
        llm_usage = {"usage": None}
        final_answer = ""

        # 3. 迭代执行循环
        while function_call_state and iteration_step <= max_iteration_steps:
            function_call_state = False
            
            # 3.1 创建 Agent 思考记录
            agent_thought_id = self.create_agent_thought(
                message_id=message.id, message="", tool_name="", 
                tool_input="", messages_ids=[]
            )

            # 3.2 重新计算 LLM 最大 token 数
            prompt_messages = self._organize_prompt_messages()
            self.recalc_llm_max_tokens(self.model_config, prompt_messages)
            
            # 3.3 记录模型调用日志
            logging.info("=== Model Invocation - Iteration %d ===", iteration_step)
            logging.info("Prompt Messages Count: %d", len(prompt_messages))
            logging.info("Available Tools: %s", [tool.name for tool in prompt_messages_tools])
            logging.info("Model Parameters: %s", app_generate_entity.model_conf.parameters)

            # 3.4 调用 LLM 模型
            chunks = model_instance.invoke_llm(
                prompt_messages=prompt_messages,
                model_parameters=app_generate_entity.model_conf.parameters,
                tools=prompt_messages_tools,
                stop=app_generate_entity.model_conf.stop,
                stream=self.stream_tool_call,
                user=self.user_id,
                callbacks=[],
            )

            # 3.5 处理模型响应
            tool_calls = []
            response = ""
            
            if isinstance(chunks, Generator):
                # 流式响应处理
                for chunk in chunks:
                    # 检查是否有工具调用
                    if self.check_tool_calls(chunk):
                        function_call_state = True
                        tool_calls.extend(self.extract_tool_calls(chunk) or [])
                    
                    # 提取响应内容
                    if chunk.delta.message and chunk.delta.message.content:
                        response += str(chunk.delta.message.content)
                    
                    yield chunk
            else:
                # 阻塞式响应处理
                result = chunks
                if self.check_blocking_tool_calls(result):
                    function_call_state = True
                    tool_calls.extend(self.extract_blocking_tool_calls(result) or [])
                
                if result.message and result.message.content:
                    response += str(result.message.content)

            # 3.6 执行工具调用
            if tool_calls:
                logging.info("=== Tool Calls - Iteration %d ===", iteration_step)
                logging.info("Number of tool calls: %d", len(tool_calls))
                
                for tool_call_id, tool_call_name, tool_call_args in tool_calls:
                    # 记录工具调用日志
                    logging.info("Tool Call ID: %s", tool_call_id)
                    logging.info("Tool Name: %s", tool_call_name)
                    logging.info("Tool Arguments: %s", tool_call_args)
                    
                    # 获取工具实例
                    tool_instance = tool_instances.get(tool_call_name)
                    if not tool_instance:
                        logging.warning("Tool not found: %s", tool_call_name)
                        continue
                    
                    # 调用工具
                    logging.info("Invoking tool: %s", tool_call_name)
                    tool_invoke_response, message_files, tool_invoke_meta = ToolEngine.agent_invoke(
                        tool=tool_instance,
                        tool_parameters=tool_call_args,
                        user_id=self.user_id,
                        tenant_id=self.tenant_id,
                        message=self.message,
                        invoke_from=self.application_generate_entity.invoke_from,
                        agent_tool_callback=self.agent_callback,
                        trace_manager=trace_manager,
                        app_id=self.application_generate_entity.app_config.app_id,
                        message_id=self.message.id,
                        conversation_id=self.conversation.id,
                    )
                    
                    # 记录工具调用结果
                    logging.info("Tool Response: %s", tool_invoke_response)
                    logging.info("Tool Meta: %s", tool_invoke_meta.to_dict())
                    
                    # 将工具响应添加到对话历史
                    self._current_thoughts.append(
                        ToolPromptMessage(
                            content=str(tool_invoke_response),
                            tool_call_id=tool_call_id,
                            name=tool_call_name,
                        )
                    )

            # 3.7 保存 Agent 思考记录
            self.save_agent_thought(
                agent_thought_id=agent_thought_id,
                tool_name=tool_call_names,
                tool_input=tool_call_inputs,
                thought=response,
                tool_invoke_meta=tool_invoke_meta,
                observation=observation,
                answer=response,
                messages_ids=message_file_ids,
                llm_usage=current_llm_usage,
            )

            iteration_step += 1

        # 4. 返回最终结果
        yield LLMResultChunk(
            model=model_instance.model,
            prompt_messages=[],
            system_fingerprint="",
            delta=LLMResultChunkDelta(
                index=0,
                message=AssistantPromptMessage(content=final_answer),
                usage=llm_usage["usage"],
            ),
        )
```

## 5. 工具执行机制

### 5.1 ToolEngine (tool_engine.py)

```python
class ToolEngine:
    @staticmethod
    def agent_invoke(
        tool: Tool,
        tool_parameters: Union[str, dict],
        user_id: str,
        tenant_id: str,
        message: Message,
        invoke_from: InvokeFrom,
        agent_tool_callback: DifyAgentCallbackHandler,
        trace_manager: Optional[TraceQueueManager] = None,
        conversation_id: Optional[str] = None,
        app_id: Optional[str] = None,
        message_id: Optional[str] = None,
    ) -> tuple[str, list[str], ToolInvokeMeta]:
        
        # 1. 参数预处理
        if isinstance(tool_parameters, str):
            # 检查工具是否只有一个参数
            parameters = [
                parameter for parameter in tool.get_runtime_parameters()
                if parameter.form == ToolParameter.ToolParameterForm.LLM
            ]
            if parameters and len(parameters) == 1:
                tool_parameters = {parameters[0].name: tool_parameters}
            else:
                # 尝试解析 JSON
                with contextlib.suppress(Exception):
                    tool_parameters = json.loads(tool_parameters)

        try:
            # 2. 触发工具开始回调
            agent_tool_callback.on_tool_start(
                tool_name=tool.entity.identity.name,
                tool_inputs=tool_parameters
            )

            # 3. 合并运行时参数
            if tool.runtime and tool.runtime.runtime_parameters:
                tool_parameters = {**tool.runtime.runtime_parameters, **tool_parameters}

            # 4. 调用工具
            response = tool.invoke(
                user_id=user_id,
                tool_parameters=tool_parameters,
                conversation_id=conversation_id,
                app_id=app_id,
                message_id=message_id,
            )

            # 5. 处理工具响应
            result = ""
            message_files = []
            
            for response_item in response:
                if isinstance(response_item, ToolInvokeMessage):
                    if response_item.type == ToolInvokeMessage.MessageType.TEXT:
                        result += response_item.message
                    elif response_item.type == ToolInvokeMessage.MessageType.JSON:
                        result += json.dumps(response_item.message, ensure_ascii=False)
                    elif response_item.type == ToolInvokeMessage.MessageType.FILE:
                        # 处理文件消息
                        message_files.append(response_item.message)

            # 6. 触发工具结束回调
            agent_tool_callback.on_tool_end(
                tool_name=tool.entity.identity.name,
                tool_inputs=tool_parameters,
                tool_outputs=result
            )

            return result, message_files, ToolInvokeMeta.success_instance()

        except Exception as e:
            # 7. 错误处理
            agent_tool_callback.on_tool_error(e)
            error_response = f"Tool {tool.entity.identity.name} invoke error: {str(e)}"
            return error_response, [], ToolInvokeMeta.error_instance(error_response)
```

### 5.2 工具类型

Dify 支持多种工具类型：

1. **BuiltinTool**: 内置工具（如 HugeGraph 查询工具）
2. **ApiTool**: API 工具
3. **WorkflowTool**: 工作流工具
4. **MCPTool**: MCP 协议工具
5. **PluginTool**: 插件工具

每种工具都继承自基础 `Tool` 类，实现 `_invoke` 方法：

```python
class Tool(ABC):
    def invoke(self, user_id: str, tool_parameters: dict[str, Any], 
              conversation_id: Optional[str] = None,
              app_id: Optional[str] = None,
              message_id: Optional[str] = None) -> Generator[ToolInvokeMessage]:
        
        # 1. 合并运行时参数
        if self.runtime and self.runtime.runtime_parameters:
            tool_parameters.update(self.runtime.runtime_parameters)

        # 2. 转换参数类型
        tool_parameters = self._transform_tool_parameters_type(tool_parameters)

        # 3. 调用具体实现
        result = self._invoke(
            user_id=user_id,
            tool_parameters=tool_parameters,
            conversation_id=conversation_id,
            app_id=app_id,
            message_id=message_id,
        )

        # 4. 处理返回结果
        if isinstance(result, ToolInvokeMessage):
            def single_generator():
                yield result
            return single_generator()
        elif isinstance(result, list):
            def generator():
                yield from result
            return generator()
        else:
            return result

    @abstractmethod
    def _invoke(self, user_id: str, tool_parameters: dict[str, Any],
               conversation_id: Optional[str] = None,
               app_id: Optional[str] = None,
               message_id: Optional[str] = None) -> ToolInvokeMessage | list[ToolInvokeMessage] | Generator[ToolInvokeMessage, None, None]:
        pass
```

## 6. 关键数据流

### 6.1 请求数据流

```
用户输入 → Controller 解析 → Service 验证 → Generator 创建实体 → Runner 执行
```

**关键数据结构：**

1. **AgentChatAppGenerateEntity**: 包含所有生成所需的信息
   - `task_id`: 任务唯一标识
   - `app_config`: 应用配置
   - `model_conf`: 模型配置
   - `inputs`: 用户输入变量
   - `query`: 用户查询
   - `files`: 上传文件
   - `user_id`: 用户ID
   - `stream`: 是否流式输出

2. **AgentEntity**: Agent 配置实体
   - `strategy`: 策略（REACT/FUNCTION_CALL）
   - `max_iteration`: 最大迭代次数
   - `tools`: 工具列表
   - `prompt`: 提示模板

### 6.2 工具调用数据流

```
LLM 输出工具调用 → 解析工具参数 → ToolEngine.agent_invoke → 具体工具._invoke → 返回结果
```

**关键数据结构：**

1. **ToolCall**: 工具调用信息
   - `id`: 调用ID
   - `name`: 工具名称
   - `arguments`: 调用参数

2. **ToolInvokeMessage**: 工具调用消息
   - `type`: 消息类型（TEXT/JSON/FILE/BLOB）
   - `message`: 消息内容
   - `meta`: 元数据

3. **ToolInvokeMeta**: 工具调用元信息
   - `time_cost`: 耗时
   - `error`: 错误信息
   - `tool_config`: 工具配置

## 7. 日志追踪机制

系统在关键节点都添加了详细的日志记录：

### 7.1 请求级别日志

```python
# 在 completion.py 中
logging.info("=== GraphAgent Chat Request ===")
logging.info("App ID: %s", app_model.id)
logging.info("App Mode: %s", app_model.mode)
logging.info("Request Args: %s", {
    "inputs": args.get("inputs"),
    "query": args.get("query"),
    "files": args.get("files"),
    "model_config": args.get("model_config"),
    "conversation_id": args.get("conversation_id"),
    "response_mode": args.get("response_mode")
})
```

### 7.2 Agent 执行日志

```python
# 在 app_runner.py 中
logging.info("Agent Max Iteration: %s", agent_entity.max_iteration)
logging.info("Agent Tools: %s", [tool.tool_name for tool in agent_entity.tools])
logging.info("Model Provider: %s", model_instance.provider)
logging.info("Model Name: %s", model_instance.model)
logging.info("Model Parameters: %s", application_generate_entity.model_conf.parameters)
```

### 7.3 模型调用日志

```python
# 在 fc_agent_runner.py 中
logging.info("=== Model Invocation - Iteration %d ===", iteration_step)
logging.info("Prompt Messages Count: %d", len(prompt_messages))
logging.info("Available Tools: %s", [tool.name for tool in prompt_messages_tools])
logging.info("Model Parameters: %s", app_generate_entity.model_conf.parameters)
logging.info("Stop Words: %s", app_generate_entity.model_conf.stop)
logging.info("Stream Mode: %s", self.stream_tool_call)
```

### 7.4 工具调用日志

```python
# 工具调用前
logging.info("=== Tool Calls - Iteration %d ===", iteration_step)
logging.info("Number of tool calls: %d", len(tool_calls))
logging.info("Tool Call ID: %s", tool_call_id)
logging.info("Tool Name: %s", tool_call_name)
logging.info("Tool Arguments: %s", tool_call_args)

# 工具调用后
logging.info("Tool Response: %s", tool_invoke_response)
logging.info("Tool Meta: %s", tool_invoke_meta.to_dict())
```

## 8. 错误处理机制

### 8.1 分层错误处理

1. **Controller 层**: 捕获并转换为 HTTP 错误响应
2. **Service 层**: 业务逻辑错误处理
3. **Runner 层**: Agent 执行错误处理
4. **Tool 层**: 工具调用错误处理

### 8.2 常见错误类型

```python
# 在 completion.py 中的错误处理
try:
    response = AppGenerateService.generate(...)
except services.errors.conversation.ConversationNotExistsError:
    raise NotFound("Conversation Not Exists.")
except services.errors.conversation.ConversationCompletedError:
    raise ConversationCompletedError()
except services.errors.app_model_config.AppModelConfigBrokenError:
    logging.exception("App model config broken.")
    raise AppUnavailableError()
except ProviderTokenNotInitError as ex:
    raise ProviderNotInitializeError(ex.description)
except QuotaExceededError:
    raise ProviderQuotaExceededError()
except ModelCurrentlyNotSupportError:
    raise ProviderModelCurrentlyNotSupportError()
except InvokeRateLimitError as ex:
    raise InvokeRateLimitHttpError(ex.description)
except InvokeError as e:
    raise CompletionRequestError(e.description)
except ValueError as e:
    raise e
except Exception as e:
    logging.exception("internal server error.")
    raise InternalServerError()
```

## 9. 性能优化机制

### 9.1 限流机制

1. **系统级限流**: 基于租户的日请求数限制
2. **应用级限流**: 基于应用的并发请求数限制

```python
# 系统级限流
system_rate_limiter = RateLimiter("app_daily_rate_limiter", 
                                 dify_config.APP_DAILY_RATE_LIMIT, 86400)

# 应用级限流
max_active_request = AppGenerateService._get_max_active_requests(app_model)
rate_limit = RateLimit(app_model.id, max_active_request)
```

### 9.2 流式处理

Agent 支持流式输出，提升用户体验：

```python
# 流式处理 LLM 响应
if isinstance(chunks, Generator):
    for chunk in chunks:
        if chunk.delta.message and chunk.delta.message.content:
            response += str(chunk.delta.message.content)
        yield chunk
```

### 9.3 内存管理

使用 TokenBufferMemory 管理对话历史：

```python
memory = TokenBufferMemory(conversation=conversation, model_instance=model_instance)
```

## 10. 总结

Dify Agent 的后端架构设计具有以下特点：

1. **分层清晰**: Controller → Service → Generator → Runner → Tool Engine
2. **模块化**: 每个组件职责单一，易于维护和扩展
3. **可扩展**: 支持多种 Agent 策略和工具类型
4. **可观测**: 完善的日志记录和错误处理
5. **高性能**: 限流、流式处理、内存管理等优化机制

通过这种架构设计，Dify Agent 能够高效地处理用户请求，执行复杂的工具调用，并提供良好的用户体验。

## 附录：关键文件清单

### 核心文件

1. **api/controllers/console/app/completion.py** - HTTP 请求处理
2. **api/services/app_generate_service.py** - 应用生成服务
3. **api/core/app/apps/agent_chat/app_generator.py** - Agent 聊天应用生成器
4. **api/core/app/apps/agent_chat/app_runner.py** - Agent 聊天应用运行器
5. **api/core/agent/fc_agent_runner.py** - Function Call Agent 运行器
6. **api/core/tools/tool_engine.py** - 工具执行引擎

### 配置文件

1. **GraphAgent-Hugegraph.yml** - Agent 应用配置文件

### 相关目录

1. **api/core/agent/** - Agent 相关实现
2. **api/core/tools/** - 工具相关实现
3. **api/core/app/apps/** - 应用类型实现
4. **api/controllers/** - HTTP 控制器
5. **api/services/** - 业务服务层
