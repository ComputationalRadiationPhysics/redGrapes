
#######################
    Access Demotion
#######################

A very cool feature is that functors can modify their own properties while runnig.
This allows us for example to demote resource accesses so that other functors can start earlier.
Of course the possible changes at runtime have to be well constrained (the SchedulingPolicy must implement that).
In the case of the ResourceUserPolicy it is only possible to **demote** the access, i.e. the new access has to be a subset of the old (e.g. we can change a write to read).

To do this actually, the scheduler has the `update_property` function.

.. code-block:: c++

    auto fun = scheduler.make_functor(
        [&]( MyContainer& x ) {
            // write on x
            scheduler.update_property< rmngr::ResourceUserPolicy >(
                { x.read() }    
            )
            // now only read x
	},
	[&]( Scheduler::SchedulablePtr s, MyContainer& x ) {
            s->proto_propery< rmngr::ResourceUserPolicy >()
                .access_list = { x.write() };
	}
    );
