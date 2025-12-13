# GraphAgent 调用关键日志输出

## 概述

针对GraphAgent (对应DSL: GraphAgent-Hugegraph.yml) 的chat调用流程，我在关键位置添加了详细的日志输出，用于追踪：

1. **web传递到后端的参数**
2. **调用模型插件传递的参数**

## 调用流程分析

当点击chat后，页面会调用 `POST /console/api/apps/6c121b20-e486-497c-bf61-8cb5bba02279/chat-messages`

### 1. 请求入口 - ChatMessageApi

**文件**: `api/controllers/console/app/completion.py`
**位置**: `ChatMessageApi.post()` 方法

**添加的日志**:
```python
# 日志1: web传递到后端的参数
logging.info("=== GraphAgent Chat Request ===")
logging.info("App ID: %s", app_model.id)
logging.info("App Mode: %s", app_model.mode)
logging.info("Request Args: %s", {
    "inputs": args.get("inputs"),
    "query": args.get("query"),
    "files": args.get("files"),
    "model_config": args.get("model_config"),
    "conversation_id": args.get("conversation_id"),
    "parent_message_id": args.get("parent_message_id"),
    "response_mode": args.get("response_mode"),
    "retriever_from": args.get("retriever_from")
})
```

**输出内容**:
- App ID 和模式
- 完整的请求参数，包括用户输入、查询、文件、模型配置等

### 2. Agent配置 - AgentChatAppRunner

**文件**: `api/core/app/apps/agent_chat/app_runner.py`
**位置**: `AgentChatAppRunner.run()` 方法

**添加的日志**:
```python
# 日志2: Agent配置和模型参数
logging.info("=== Agent Configuration ===")
logging.info("Agent Strategy: %s", agent_entity.strategy)
logging.info("Agent Max Iteration: %s", agent_entity.max_iteration)
logging.info("Agent Tools: %s", [tool.tool_name for tool in agent_entity.tools] if agent_entity.tools else [])
logging.info("Model Provider: %s", model_instance.provider)
logging.info("Model Name: %s", model_instance.model)
logging.info("Model Parameters: %s", application_generate_entity.model_conf.parameters)
```

**输出内容**:
- Agent策略 (FUNCTION_CALLING/CHAIN_OF_THOUGHT)
- 最大迭代次数
- 可用工具列表
- 模型提供商和名称
- 模型参数

### 3. 模型调用 - FunctionCallAgentRunner

**文件**: `api/core/agent/fc_agent_runner.py`
**位置**: `FunctionCallAgentRunner.run()` 方法

**添加的日志**:
```python
# 日志3: 调用模型前的参数
logging.info("=== Model Invocation - Iteration %d ===", iteration_step)
logging.info("Prompt Messages Count: %d", len(prompt_messages))
logging.info("Available Tools: %s", [tool.name for tool in prompt_messages_tools] if prompt_messages_tools else [])
logging.info("Model Parameters: %s", app_generate_entity.model_conf.parameters)
logging.info("Stop Words: %s", app_generate_entity.model_conf.stop)
logging.info("Stream Mode: %s", self.stream_tool_call)
```

**输出内容**:
- 当前迭代轮次
- Prompt消息数量
- 可用工具列表
- 模型参数
- 停止词
- 流式模式

### 4. 工具调用 - FunctionCallAgentRunner

**文件**: `api/core/agent/fc_agent_runner.py`
**位置**: 工具调用循环中

**添加的日志**:
```python
# 日志4: 工具调用参数
logging.info("=== Tool Calls - Iteration %d ===", iteration_step)
logging.info("Number of tool calls: %d", len(tool_calls))
logging.info("Tool Call ID: %s", tool_call_id)
logging.info("Tool Name: %s", tool_call_name)
logging.info("Tool Arguments: %s", tool_call_args)

# 日志5: 工具调用结果
logging.info("Invoking tool: %s with parameters: %s", tool_call_name, tool_call_args)
logging.info("Tool Response: %s", tool_invoke_response)
logging.info("Tool Meta: %s", tool_invoke_meta.to_dict() if tool_invoke_meta else None)
```

**输出内容**:
- 工具调用数量
- 每个工具的调用ID、名称和参数
- 工具调用结果和元数据

## 日志查看方式

1. **日志文件位置**: `api/logs/output.log`
2. **实时查看**: `tail -f api/logs/output.log`
3. **过滤GraphAgent日志**: `grep "GraphAgent\|Agent Configuration\|Model Invocation\|Tool Calls" api/logs/output.log`

## 示例日志输出

```
INFO - === GraphAgent Chat Request ===
INFO - App ID: 6c121b20-e486-497c-bf61-8cb5bba02279
INFO - App Mode: agent-chat
INFO - Request Args: {'inputs': {'user_question': '请分析用户user_4的2跳关系', 'user_context': '', 'answer_style': '简洁说明'}, 'query': '请分析用户user_4的2跳关系', 'files': [], 'model_config': {...}, ...}

INFO - === Agent Configuration ===
INFO - Agent Strategy: AgentEntity.Strategy.FUNCTION_CALLING
INFO - Agent Max Iteration: 10
INFO - Agent Tools: ['hugegraph_query', 'hugegraph_gremlin']
INFO - Model Provider: langgenius/openai/openai
INFO - Model Name: gpt-5-chat-latest
INFO - Model Parameters: {'temperature': 0.7, 'top_p': 1.0, ...}

INFO - === Model Invocation - Iteration 1 ===
INFO - Prompt Messages Count: 3
INFO - Available Tools: ['hugegraph_query', 'hugegraph_gremlin']
INFO - Model Parameters: {'temperature': 0.7, 'top_p': 1.0, ...}
INFO - Stop Words: []
INFO - Stream Mode: True

INFO - === Tool Calls - Iteration 1 ===
INFO - Number of tool calls: 1
INFO - Tool Call ID: call_123456
INFO - Tool Name: hugegraph_gremlin
INFO - Tool Arguments: {'gremlin_query': "g.V('user_4').repeat(both().simplePath()).times(2).valueMap(true).limit(50)"}
INFO - Invoking tool: hugegraph_gremlin with parameters: {'gremlin_query': "g.V('user_4').repeat(both().simplePath()).times(2).valueMap(true).limit(50)"}
INFO - Tool Response: [查询结果...]
INFO - Tool Meta: {'time_cost': 0.5, 'error': None, ...}
```

## 关键信息说明

1. **web传递参数**: 可以看到前端传递的完整参数，包括用户输入的问题、上下文、回答风格等
2. **模型配置**: 显示使用的模型提供商、模型名称、参数配置
3. **工具配置**: 显示Agent可用的工具列表 (HugeGraph查询工具)
4. **调用过程**: 每次模型调用和工具调用的详细参数和结果

这些日志可以帮助你完整追踪GraphAgent从接收请求到调用HugeGraph工具的整个流程。
