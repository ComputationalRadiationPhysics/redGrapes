/* Copyright 2019 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <unordered_map>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/context/continuation.hpp>
#include <akrzemi/optional.hpp>
#include <redGrapes/graph/scheduling_graph.hpp>
#include <redGrapes/graph/util.hpp>

#include <vector>

namespace redGrapes
{

template <
    typename TaskID,
    typename TaskPtr,
    typename PrecedenceGraph
>
struct SchedulerBase
{
    using EventID = typename SchedulingGraph< TaskID, TaskPtr >::EventID;

    struct Job
    {
        std::shared_ptr< TaskImplBase > f;
        TaskPtr task_ptr;

        void operator() ()
        {
            (*f)();
        }

        void yield( EventID event_id )
        {
            f->yield( event_id );
        }
    };

    std::shared_ptr< PrecedenceGraph > precedence_graph;
    SchedulingGraph< TaskID, TaskPtr > scheduling_graph;
    std::vector< ThreadSchedule<Job> > schedule;

    std::atomic_flag uptodate;
    std::atomic_bool finishing;

    SchedulerBase( std::shared_ptr<PrecedenceGraph> precedence_graph, size_t n_threads )
        : precedence_graph( precedence_graph )
        , schedule( n_threads + 1 )
        , finishing( false )
    {
        uptodate.clear();
    }

    void notify()
    {
        uptodate.clear();
        for( auto & thread : schedule )
            thread.notify();
    }

    void finish()
    {
        finishing = true;
        notify();
    }

    void update_vertex( TaskPtr p )
    {
        scheduling_graph.update_vertex( p );
        notify();
    }

    void reach_event( EventID event_id )
    {
        scheduling_graph.reach_event( event_id );
        notify();
    }

    /*
     * pause the current task until event_id is reached
     */
    void yield( EventID event_id )
    {
        if( thread::id >= schedule.size() )
            return;

        if( std::experimental::optional<Job> job = schedule[ thread::id ].get_current_job() )
        {
            job->yield( event_id );
        }
        else
        {
            (*this)(
                [this, event_id]
                {
                    return this->scheduling_graph.is_event_reached( event_id );
                }
            );
        }
    }

    /*
     * consume jobs until pred returns true
     * after finish() was called it returns whenever all tasks have finished
     */
    void operator() ( std::function<bool()> const & pred = []{ return false; } )
    {
        auto stop =
            [this, pred]
            {
                return (finishing && scheduling_graph.empty()) || pred();
            };

        while( ! stop() )
            schedule[ thread::id ].consume( stop );
    }

    std::experimental::optional<TaskPtr> get_current_task()
    {
        if( thread::id >= schedule.size() )
            return std::experimental::nullopt;

        if( std::experimental::optional<Job> job = schedule[ thread::id ].get_current_job() )
            return std::experimental::optional<TaskPtr>( job->task_ptr );
        else
            return std::experimental::nullopt;
    }
};

} // namespace redGrapes

