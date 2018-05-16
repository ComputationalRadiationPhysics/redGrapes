

#include <rmngr/scheduling_context.hpp>

int fun1_impl (int x)
{
    return x*x;
}

int main()
{
    rmngr::SchedulingContext context( 1 /* number of threads */ );
    auto queue = context.get_main_queue();

    auto fun1_proto = context.make_proto( &fun1_impl );

    // set flags (optional, they all have defaults)
    fun1_proto.label = "Functor 1";
    fun1_proto.main_thread = false;

    // create object which pushes a clone of proto into the queue
    auto fun1 = queue.make_functor( fun1_proto );


    // call fun1 like fun1_impl
    int i = 4;
    std::future<int> res = fun1( i );
    std::cout << "result: " << res.get() << std::endl;

    return 0;
}

