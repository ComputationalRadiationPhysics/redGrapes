
#include <iostream>
#include <thread>
#include <chrono>

#include <rmngr/resource/fieldresource.hpp>
#include <rmngr/scheduler/scheduler_singleton.hpp>
#include <rmngr/scheduler/resource.hpp>
#include <rmngr/scheduler/dispatch.hpp>
#include <rmngr/scheduler/fifo.hpp>

using Scheduler = rmngr::SchedulerSingleton<
    boost::mpl::vector<
        rmngr::ResourceUserPolicy,
        rmngr::DispatchPolicy< rmngr::FIFO >
    >
>;

struct Buffer : rmngr::FieldResource<1>
{
    void do_something( )
    {
        Scheduler::enqueue_functor(
            []()
            {
                std::cout << "read buffer" << std::endl;
                std::this_thread::sleep_for( std::chrono::seconds(1) );
            },
            [this]( rmngr::observer_ptr<Scheduler::Schedulable> s )
            {
                s->proto_property< rmngr::ResourceUserPolicy >().access_list =
                {
                    this->FieldResource<1>::read()
                };
            }
        );
    }
};

int main( int, char*[] )
{
    Scheduler::init(8);

    Buffer b;
    b.do_something();
    b.do_something();

    Scheduler::enqueue_functor(
        [ &b ]()
        {
            std::cout << "hello" << std::endl;
        },
        [ &b ]( rmngr::observer_ptr<Scheduler::Schedulable> s )
        {
            s->proto_property< rmngr::ResourceUserPolicy >().access_list =
            {
               b.write()
            };
        }
    );

    Scheduler::finish();

    return 0;
}

