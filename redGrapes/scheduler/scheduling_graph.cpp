/* Copyright 2019-2020 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

namespace redGrapes
{
namespace scheduler
{

SchedulingGraphProp::SchedulingGraphProp()
{}

SchedulingGraphProp::SchedulingGraphProp(SchedulingGraphProp const & other)
{
    
}

bool SchedulingGraphProp::is_ready() { return pre_event.is_ready(); }
bool SchedulingGraphProp::is_running() { return pre_event.is_reached(); }
bool SchedulingGraphProp::is_finished() { return post_event.is_reached(); }

/*! create a new event which precedes the tasks post-event
 */
EventPtr SchedulingGraphProp::make_event()
{
        /*
        Event event = std::make_shared< Event >();
        event.add_follower( post_event );
        return event;
        */        
}

/*!
 * represent ›pausation of the task until event is reached‹
 * in the scheduling graph
 *
void SchedulingGraphProp::sg_pause( EventPtr event )
{
    auto task_vertex = pre_event.task_vertex;

    pre_event.state = 1;

    //SPDLOG_TRACE("sg pause: new_event = {}", (void*) event.get());
    event->add_follower(EventPtr{ T_PRE_EVENT, task_vertex });
}
*/
/*!
 * Insert a new task and add the same dependencies as in the precedence graph.
 * Note that tasks must be added in order, since only preceding tasks are considered!
 *
 * The precedence graph containing the task is assumed to be locked.
 */
template < typename Task, typename RedGrapes >
void SchedulingGraphProp::sg_init( RedGrapes & rg, TaskVertexPtr task_vertex )
{
    SPDLOG_TRACE("sg init task {}", task_vertex->template get_task<Task>().task_id);

    // add dependencies to tasks which precede the new one
    for(auto weak_in_vertex_ptr : task_vertex->in_edges)
    {
        if( auto preceding_task_vertex = weak_in_vertex_ptr.lock() )
        {
            auto & preceding_task = preceding_task_vertex->template get_task<Task>();
            auto preceding_event = EventPtr {
                rg.get_scheduler()->task_dependency_type(preceding_task_vertex, task_vertex) ? T_EVT_PRE : T_EVT_POST, preceding_task_vertex
            };

            if(! preceding_event->is_reached() )
                preceding_event->add_follower(EventPtr{ T_EVT_PRE, task_vertex });
        }
    }

    // add dependency to parent
    if( auto parent = task_vertex->space.lock()->parent )
        post_event.add_follower( EventPtr{ T_EVT_POST, parent->lock() } );
}

/*! remove revoked dependencies (e.g. after access demotion)
 *
 * @param revoked_followers set of tasks following this task
 *                          whose dependency on it got removed
 *
 * The precedence graph containing task_vertex is assumed to be locked.
 */
template < typename Task, typename RedGrapes >
void SchedulingGraphProp::sg_revoke_followers( RedGrapes & rg, TaskVertexPtr task_vertex, std::vector<TaskVertexPtr> revoked_followers )
{
    for( auto follower_vertex : revoked_followers )
    {
        if( ! rg.get_scheduler()->task_dependency_type( task_vertex, follower_vertex ) )
        {
            auto event = EventPtr{ T_EVT_PRE, follower_vertex };
            post_event.remove_follower( event );
            rg.notify_event( event );
        }
        // else: the pre-event of task_vertex's task shouldn't exist at this point, so we do nothing
   }
}


} // namespace scheduler

} // namespace redGrapes


