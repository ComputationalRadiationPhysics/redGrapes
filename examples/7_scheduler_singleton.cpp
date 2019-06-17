
#include <iostream>
#include <thread>
#include <chrono>

#include <rmngr/resource/fieldresource.hpp>
#include <rmngr/scheduler/scheduler_singleton.hpp>
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
    rmngr::SchedulerSingleton<
        boost::mpl::vector<
            rmngr::ResourceUserPolicy,
            rmngr::DispatchPolicy< rmngr::FIFO >
        >,
        PrecedenceGraph
    >;

struct Buffer : rmngr::FieldResource<1>
{
    void do_something( )
    {
        Scheduler::Properties prop;
        prop.policy<rmngr::ResourceUserPolicy>() += this->FieldResource<1>::read();

        Scheduler::emplace_task(
            []
            {
                std::cout << "read buffer" << std::endl;
                std::this_thread::sleep_for( std::chrono::seconds(1) );
            },
            prop
        );
    }
};

int main( int, char*[] )
{
    Scheduler::init(8);

    Buffer b;
    b.do_something();
    b.do_something();

    Scheduler::Properties prop;
    prop.policy< rmngr::ResourceUserPolicy >() += b.write();
    
    Scheduler::emplace_task(
        [ &b ]
        {
            std::cout << "hello" << std::endl;
        },
        prop
    );

    Scheduler::finish();

    return 0;
}

