# API notes (verified against the installed packages)

Installed versions at the time of writing: **langgraph 1.2.20**, **langgraph-checkpoint 4.3.1**, **langgraph-checkpoint-sqlite 3.1.3**, **langchain 1.4.9**, **langchain-core 1.7.6**. Every fact below is also a test in `tests/test_api_facts.py`; the test name is given in brackets. Facts were checked by introspecting signatures and reading the installed source, not from memory.

| # | Fact | Where verified |
|---|---|---|
| F1 | `Send(node, arg)` targets listed by a conditional edge all run in one superstep; a node with a plain edge from the target runs **once** afterwards (fan-in). `add_node(..., input_schema=Payload)` gives the target its own input type. | `langgraph.types.Send.__init__`, `StateGraph.add_node` signature [`test_send_fan_out_and_fan_in`] |
| F2 | A node may return `Command(goto=..., update=...)`; `add_node(..., destinations=(...))` makes those edges appear in `get_graph().draw_mermaid()`. | `Command` fields `graph, update, resume, goto` [`test_command_goto_and_destinations_in_mermaid`] |
| F3 | `interrupt(value)` inside a node makes `invoke` return `{"__interrupt__": [Interrupt(value=..., id=...)]}`; `invoke(Command(resume=v), config)` re-runs the node from its start and `interrupt()` returns `v`. The `configurable` dict passed at resume time is what the node sees. `Interrupt` has `id`, not the pre-1.0 `ns`. | `langgraph.types.interrupt`, `Interrupt.__dataclass_fields__` [`test_interrupt_resume_and_configurable_on_resume`] |
| F4 | `recursion_limit` counts supersteps: the minimal limit is **supersteps + 1** (measured: a loop of `n` single-node iterations needs `n + 1`; a `prepare → sample(Send ×K) → estimate → control` iteration costs 4 supersteps, so `n` controlled steps need `4n + 1`). `GraphRecursionError` lives in `langgraph.errors`. | [`test_recursion_limit_counts_supersteps`] |
| F5 | `get_state_history(config)` yields `StateSnapshot`s newest first with `.values`, `.next`, `.config`; `update_state(snapshot.config, values, as_node=name)` forks from that checkpoint and returns the forked config; `invoke(None, forked_config)` continues from there under whatever `configurable` is passed then. The original tip remains in history. | `Pregel.update_state`, `Pregel.get_state_history` signatures [`test_checkpoint_forking_with_update_state`] |
| F6 | Pydantic models stored in state survive the checkpoint round trip across an interrupt (JsonPlus serializer). | [`test_pydantic_models_survive_checkpoint`] |
| F7 | `BaseChatModel.invoke(messages, **kwargs)` forwards kwargs to `_generate(messages, stop, run_manager, **kwargs)` (via `_generate_with_cache`). This is how `sample_index` reaches the mock without touching the prompt. | `BaseChatModel.invoke`, `_generate_with_cache` source [`test_model_kwargs_reach_generate`] |
| F8 | `InMemorySaver` (alias `MemorySaver`) in `langgraph.checkpoint.memory`; `BaseCheckpointSaver.delete_thread(thread_id)` exists; `SqliteSaver.from_conn_string(path)` is a context manager; `langchain.chat_models.init_chat_model(model, *, model_provider=None, **kwargs)`. | signatures [`test_checkpointer_and_backend_entry_points`] |
| F9 | `invoke(..., durability="exit" | "async" | "sync")` replaces the pre-1.0 `checkpoint_during`; the default is `"async"`. With `"exit"` the checkpoint is still written when the run stops at an interrupt, so resume works. | `Pregel.invoke` signature, `pregel/main.py` [`test_interrupt_resume_and_configurable_on_resume` uses `"exit"`] |
| F10 | `StateGraph.add_node` also accepts `defer=True` (run a node only when the superstep has nothing else to do); kept as the fallback if fan-in behaviour ever changes. | signature |
| F11 | `Pregel.batch(inputs, config)` exists and honours `max_concurrency`; **not used** by the runner, which uses a process pool (see `DECISIONS.md`). | signature |

## Added in M3/M4

| # | Fact | Where verified |
|---|---|---|
| F12 | `JsonPlusSerializer(allowed_msgpack_modules=[(module, name), ...])` registers custom types for checkpoint deserialisation; without it LangGraph 1.2 warns "Deserializing unregistered type" and announces a future block (`LANGGRAPH_STRICT_MSGPACK`). `InMemorySaver(serde=...)` and `SqliteSaver(conn, serde=...)` accept it. | `langgraph/checkpoint/serde/jsonplus.py` [`test_checkpoint_round_trip_emits_no_unregistered_type_warning`] |
| F13 | `BaseChatModel` has a pydantic field named `cache` (LangChain's own response cache); a subclass must not reuse the name. | `langchain_core/language_models/chat_models.py` [`tests/test_cache.py`] |
| F14 | `init_chat_model("ollama:llama3.1:8b", temperature=...)` works with `langchain-ollama 1.1.0`; the reply carries `usage_metadata` with `input_tokens`/`output_tokens`. | [`test_real_ollama_small_run_and_cached_rerun`, run 2026-11-07] |
| F15 | Forking a checkpoint taken after a node that routes with `Command(goto=...)` (no static out-edges) needs `update_state(snapshot.config, Command(update=..., goto=...), as_node=that_node)`; a plain dict update leaves `next == ()` and `invoke(None, forked)` does nothing. Used by `core/risk/influence.py`. | measured on langgraph 1.2.20 [`test_update_state_with_command_at_a_command_routing_node`] |

## Added in M9 (interactive sessions)

| # | Fact | Where verified |
|---|---|---|
| F16 | `interrupt_before=[node]` is a per-call option of `invoke`/`stream`, not only of `compile`; the static pause leaves **no** `__interrupt__` entry in `invoke`'s result and shows up as `get_state(config).next == (node,)`; `invoke(None, config, interrupt_before=...)` continues one node run further. Used by `core/graph/session.py` (supervisor mode). | measured on langgraph 1.2.20 [`test_per_call_interrupt_before_pauses_before_a_node`] |
| F17 | `stream(..., stream_mode="updates")` yields one `{node: update}` chunk per node run and ends with an `{"__interrupt__": (...)}` chunk whose tuple is **empty** for a static pause and holds `Interrupt(value, id)` for a dynamic `interrupt()`; no such chunk when the run ends. A one-off `configurable` for a single call (the human override) followed by a resume under the base `configurable` behaves as F3 says. | measured on langgraph 1.2.20 [`test_stream_updates_surfaces_static_and_dynamic_pauses`] |

## Added in M12 (provider options)

| # | Fact | Where verified |
|---|---|---|
| F18 | `init_chat_model(name, temperature=..., **kwargs)` forwards unknown kwargs to the provider class. For `ChatOllama` 1.1.0, `reasoning="low"` is sent as Ollama's `think` level and the model's thinking comes back in `additional_kwargs["reasoning_content"]`, never in `content` (80 of 80 cached `gpt-oss:20b` replies hold only the `Answer:` and `Confidence:` lines; output at most 212 tokens under a 1024 cap); `num_ctx` and `num_predict` travel in `options`, and `ollama ps` shows the per-request context (8192) and 100 % GPU placement where the server default (49,152) had forced a CPU/GPU split. A `reasoning` kwarg on a model without thinking support makes Ollama return an error, so options are per config. | measured on langchain-ollama 1.1.0 and Ollama 0.15.2 (smoke run of 2026-11-07, `configs/musique_smoke_gptoss.yaml`) [`test_backend_uses_init_chat_model`, `test_model_kwargs_in_key_only_when_present`] |
