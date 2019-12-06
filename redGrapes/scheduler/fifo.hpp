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
    typename TaskID,
    typename TaskPtr,
    typename PrecedenceGraph
>
struct FIFOScheduler
    : StateScheduler< TaskID, TaskPtr, PrecedenceGraph >
{
    using typename StateScheduler<TaskID, TaskPtr, PrecedenceGraph>::Job;

    FIFOScheduler( std::shared_ptr<PrecedenceGraph> & pg, size_t n_threads )
        : StateScheduler< TaskID, TaskPtr, PrecedenceGraph >(  pg, n_threads )
    {
        for( auto & t : this->schedule )
            t.set_request_hook( [this,&t]{ get_job(t); } );
    }

private:
    std::mutex queue_mutex;
    std::queue< Job > job_queue;

    void get_job( ThreadSchedule< Job > & thread )
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
