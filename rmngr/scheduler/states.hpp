
#pragma once

#include <unordered_map>

namespace rmngr
{

enum TaskState { pending = 0, ready, scheduled, running, done };

template < typename TaskID >
struct TaskStateMap
    : std::unordered_map< TaskID, TaskState >
{
    void prepare_task_states( TaskID task )
    {
        task->hook_before( [this, task] { (*this)[ task ] = TaskState::running; } );
        task->hook_after( [this, task] { (*this)[ task ] = TaskState::done; } );
        (*this)[ task ] = TaskState::pending;
    }
};

} // namespace rmngr

