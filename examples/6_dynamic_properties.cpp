
#include <iostream>
#include <thread>
#include <chrono>

#include <rmngr/resource/fieldresource.hpp>
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/scheduler/resource.hpp>
#include <rmngr/scheduler/dispatch.hpp>
#include <rmngr/scheduler/fifo.hpp>

template <typename Graph>
using PrecedenceGraph =
    rmngr::QueuedPrecedenceGraph<
        Graph,
        rmngr::ResourceEnqueuePolicy
    >;

using Scheduler =
    rmngr::Scheduler<
        boost::mpl::vector<
            rmngr::ResourceUserPolicy,
            rmngr::DispatchPolicy< rmngr::FIFO >
        >,
        PrecedenceGraph
    >;

int main( int, char*[] )
{
    Scheduler scheduler(4);

    rmngr::FieldResource<1> field;

    auto fun1 = scheduler.make_functor(
        []( int x )
        {
            std::cout << "Access " << x << std::endl;
            std::this_thread::sleep_for( std::chrono::seconds(1) );            
        },
        [field]( int x )
        {
            Scheduler::Properties prop;
            prop.policy<rmngr::ResourceUserPolicy>() += field.write({{x,x}});
            return prop;
        }
    );

    fun1( 1 );
    fun1( 2 );
    fun1( 3 );

    fun1( 1 );
    fun1( 2 );
    fun1( 4 );

    return 0;
}

