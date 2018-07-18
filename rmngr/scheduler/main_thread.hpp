
#pragma once

#include <rmngr/scheduler/dispatch.hpp> /* DefaultJobSelector */

namespace rmngr
{

template <
    typename Job,
    template <typename> class JobSelector
>
struct MainThreadSelector : DefaultJobSelector<Job>
{
    JobSelector<Job> main_sel, others_sel;

    struct Property : JobSelector<Job>::Property
    {
        Property() : main_thread(false) {}
        bool main_thread;
    };

    bool empty( void )
    {
        return main_sel.empty() && others_sel.empty();
    }

    void push( Job const & j, Property const & prop = Property() )
    {
        if( prop.main_thread )
            this->main_sel.push( j, prop );
        else
            this->others_sel.push( j, prop );
    }

    Job getJob( void )
    {
        if( rmngr::thread::id == 0 )
            return main_sel.getJob();
        else
            return others_sel.getJob();
    }
};

} // namespace rmngr

