
#include <rmngr/scheduler/scheduler.hpp>
#include <rmngr/scheduler/resource.hpp>
#include <rmngr/scheduler/dispatch.hpp>
#include <rmngr/scheduler/fifo.hpp>
#include <rmngr/scheduler/graphviz.hpp>

#include <rmngr/resource/ioresource.hpp>

int fun1_impl (int x)
{
    return x*x;
}

int main()
{
    using Scheduler = rmngr::Scheduler<
        boost::mpl::vector<
            rmngr::DispatchPolicy< rmngr::FIFO >
        >
    >;

    Scheduler scheduler( 0 /* number of threads */ );

    auto fun = scheduler.make_functor(&fun1_impl);
    std::cout << "fun(2) = " << fun(2).get() << std::endl;

    return 0;
}

