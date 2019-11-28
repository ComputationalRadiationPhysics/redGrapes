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
    typename SchedulingGraph,
    typename PrecedenceGraph
>
struct FIFOScheduler
    : StateScheduler< SchedulingGraph, PrecedenceGraph >
{
    using TaskID = typename SchedulingGraph::TaskID;
    using TaskPtr = typename SchedulingGraph::TaskPtr;
    using Job = typename SchedulingGraph::Job;

    std::mutex queue_mutex;
    std::queue< Job > job_queue;

    FIFOScheduler( std::unordered_map<TaskID, TaskPtr> & tasks, std::shared_mutex & m, SchedulingGraph & graph, std::shared_ptr<PrecedenceGraph> & pg )
        : StateScheduler< SchedulingGraph, PrecedenceGraph >( tasks, m, graph, pg )
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
            if( ! this->uptodate.test_and_set() )
            {
                auto ready_tasks = this->update_graph();
                for( auto job : ready_tasks )
                    job_queue.push( job );
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
