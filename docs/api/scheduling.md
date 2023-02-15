# Scheduler
``` mermaid
sequenceDiagram
  participant U as User
  participant S as Scheduler
  participant E as Executor

  activate U
  U-->>S: on( STARTED, callback )
  U-->>S: on( FINISHED, callback )

  U->>S: run(...)
  deactivate U

  activate S
  loop EventLoop
    note left of S: Event: STARTED
    par Task
      S->>E: execute(f, *args, **kwargs)
      activate E
      note left of S: Event: SUBMITTED
      E-->>S: _process_future(...)
      note left of S: Event: {FINISHED, COMPLETE, ERROR, CANCELLED}
      deactivate E
    and Parallel Task
      S->>E: execute(f, *args, **kwargs)
      activate E
      note left of S: Event: SUBMITTED
      E-->>S: _process_future(...)
      note left of S: Event: {FINISHED, COMPLETE, ERROR, CANCELLED}
      deactivate E
    end

    break when stopping criterion met
      S-->E: {Timeout, Stop, Empty}
      note left of S: Event: STOPPING
      S->>E: executor.shutdown(...)
      note left of S: Event: FINISHED
    end
  end
  S->>U: ExitCode: {TIMEOUT, STOPPED, EMPTY, UNKNOWN}
  deactivate S
```

::: byop.scheduler.scheduler
