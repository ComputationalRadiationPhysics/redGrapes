
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/scheduler/resource.hpp>
#include <rmngr/scheduler/dispatch.hpp>
#include <rmngr/scheduler/fifo.hpp>

int fun1_impl (int x)
{
    return x*x;
}

int main()
{
    rmngr::Scheduler<
        boost::mpl::vector<
            rmngr::ResourceUserPolicy,
            rmngr::DispatchPolicy< rmngr::FIFO >
        >
    > scheduler( 1 /* number of threads */ );

    auto queue = scheduler.get_main_queue();
    auto fun1_proto = scheduler.make_proto( &fun1_impl );

    // create object which pushes a clone of proto into the queue
    auto fun1 = queue.make_functor( fun1_proto );

    // call fun1 like fun1_impl
    int i = 4;
    std::future<int> res = fun1( i );
    std::cout << "result: " << res.get() << std::endl;

    return 0;
}

