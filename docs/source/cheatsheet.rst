Cheatsheet
==========

General
-------

- Getting redGrapes: https://github.com/ComputationalRadiationPhysics/redGrapes
- Issue tracker, questions, support: https://github.com/ComputationalRadiationPhysics/redGrapes/issues

Header Files
------------

- ``redGrapes/manager.hpp``: Essential manager class
- ``redGrapes/property/``: Policies defining additional task properties
- ``redGrapes/access/``: Resource-Access policies
- ``redGrapes/resource/``: Resource containers with safety wrappers
- ``redGrapes/helpers/cupla/scheduler.hpp``: Scheduler implementation for asynchronous cupla tasks.
- ``redGrapes/helpers/mpi/request_pool.hpp``: Helper to wait for ``MPI_Requests`` without blocking. Provides an adapter
  from MPI_Requests to redGrapes events.
- ``redGrapes/helpers/mpi/scheduler.hpp``: Default MPI-Scheduler using a FIFO and providing a convenience wrapper to create MPI tasks, execution is user controlled.


In further code, the following is assumed in addition to the appropriate includes


  .. code-block:: c++

    using namespace rg = redGrapes;

Initialization
-----------

  .. code-block:: c++

    rg::Manager<
        /* variadic template to configure
	 * task-properties. Can be empty.
	 * IDProperty and ResourceProperty
	 * are included always and should
	 * not be in this list.
         */
        rg::LabelProperty  // optional, useful debug info
	// .. more
    > mgr;

    // used to construct the properties for each task.
    using TaskProperties = decltype(mgr)::TaskProps;

Basic Task Operations
---------------------
    
Declare Resources
  .. code-block:: c++

    rg::IOResource< int > res1;

Create Tasks
  .. code-block:: c++

                                /* optional */ /* optional */  
    mgr.emplace_task( Callable, TaskProperties, Resources ... );

Task Properties
  .. code-block:: c++

    TaskProperties::Builder()
        .label("Label") /* requires rg::LabelProperty */

Get Task Results
  .. code-block:: c++

    auto result = mgr.emplace_task( ... ).get();


Full Task Creation
  .. code-block:: c++

    auto fut = mgr.emplace_task(
        [] ( auto r1 )
	{
	    return (*r1) * (*r1);
	},
	TaskProperties::Builder().label("Task 1"),
	res1.read()
    );


Events
------

Create Event (current task will only finish after this event was flagged)

  .. code-block:: c++

    auto event_id = mgr.create_event();

Flag Event

  .. code-block:: c++

    mgr.reach_event( event_id );
    
Access Policies
---------------

An Access Policy satisfies the following concept:

.. code-block:: c++

    struct MyAccess
    {
        static bool is_serial( MyAccess const & a, MyAccess const & b ) const;
	bool is_superset_of( AreaAccess const & a ) const;
    };

    
Configure Scheduler
-------------------

  .. code-block:: c++

    rg::Manager<
        rg::LabelProperty,
        rg::helpers::cupla::CuplaTaskProperties
    > mgr;

    auto cupla_scheduler = rg::helpers::cupla::make_cupla_scheduler( 8 /* optional, number of cupla streams */ );
    auto default_scheduler = rg::scheduler::make_default_scheduler( 8 /* optional, number of CPU threads */ );

    mgr.set_scheduler(
        rg::scheduler::make_tag_match_scheduler( mgr )
	    // all tasks with the SCHED_CUPLA tag are scheduled by cupla_scheduler
            .add({ SCHED_CUPLA }, cupla_scheduler )
	    // default case
	    .add({}, default_scheduler )
    );
