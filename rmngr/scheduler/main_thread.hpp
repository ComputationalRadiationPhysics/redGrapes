
#pragma once

#include <rmngr/scheduler/fifo.hpp>

namespace rmngr
{

template <
    typename Executable,
    template <typename> class JobSelector = FIFO
>
struct MainThreadDispatcher
{
    JobSelector<Executable> main_sel, others_sel;

    struct Property : JobSelector<Executable>::Property
    {
        Property() : main_thread(false) {}
        bool main_thread;
    };

    bool empty( void )
    {
        return main_sel.empty() && others_sel.empty();
    }

    void push( Executable e, Property & prop )
    {
        if( prop.main_thread )
            this->main_sel.push( e, prop );
        else
            this->others_sel.push( e, prop );
    }

    Executable getJob( void )
    {
        if( rmngr::thread::id == 0 )
            return main_sel.getJob();
        else
            return others_sel.getJob();
    }
};

} // namespace rmngr

