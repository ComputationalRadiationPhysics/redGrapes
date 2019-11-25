/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <mutex>
#include <queue>
#include <redGrapes/scheduler/states.hpp>

namespace redGrapes
{

template <
    typename TaskProperties,
    typename SchedulingGraph
>
struct FIFOScheduler
    : StateScheduler< TaskProperties, SchedulingGraph >
{
    using TaskID = typename TaskContainer<TaskProperties>::TaskID;
    using Job = typename SchedulingGraph::Job;

    std::mutex queue_mutex;
    std::queue< Job > job_queue;

    FIFOScheduler( TaskContainer< TaskProperties > & tasks, SchedulingGraph & graph )
        : StateScheduler< TaskProperties, SchedulingGraph >( tasks, graph )
    {
        for( auto & t : this->graph.schedule )
            t.set_request_hook( [this,&t]{ get_job(t); } );
    }

private:
    void get_job( typename SchedulingGraph::ThreadSchedule & thread )
    {
        std::unique_lock< std::mutex > lock( queue_mutex );

        if( job_queue.empty() )
        {
            bool u1 = this->uptodate.test_and_set();
            bool u2 = this->graph.precedence_graph.test_and_set();
            if( !u1 || !u2 )
            {
                auto ready_tasks = this->update_graph();
                for( TaskID t : ready_tasks )
                    job_queue.push( Job{ this->tasks, t } );
            }
        }

        if( ! job_queue.empty() )
        {
            auto job = job_queue.front();
            job_queue.pop();
            thread.push( job );
        }
    }

};

} // namespace redGrapes
