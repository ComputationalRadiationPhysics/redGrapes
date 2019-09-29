
#pragma once

#include <mutex>
#include <functional>

namespace rmngr
{

template < typename TaskProperties >
struct Task
{
    Task( TaskProperties const & properties )
        : properties( properties )
        , before_hook( []{} )
        , after_hook( []{} )
        , done( false )
    {}

    virtual ~Task()
    {
        std::unique_lock< std::mutex > lock( cv_mutex );
        cv.wait( lock, [this]{ return done; } );
    }

    virtual void run() = 0;

    void operator() ()
    {
        before_hook();
        run();
        after_hook();

        std::lock_guard<std::mutex> lock( cv_mutex );
        done = true;
        cv.notify_all();
    }

    void hook_before( std::function<void(void)> const & hook )
    {
        before_hook = [rest=std::move(before_hook), hook]{ hook(); rest(); };
    }

    void hook_after( std::function<void(void)> const & hook )
    {
        after_hook = [rest=std::move(after_hook), hook]{ rest(); hook(); };
    }

    std::mutex cv_mutex;
    std::condition_variable cv;
    bool done;

    std::function<void(void)> before_hook, after_hook;
    TaskProperties properties;
};

template< typename TaskProperties, typename NullaryCallable >
struct FunctorTask : Task< TaskProperties >
{
    FunctorTask( NullaryCallable && impl, TaskProperties const & properties )
        : Task< TaskProperties >( properties )
        , impl( std::move(impl) )
    {}

    void run()
    {
        this->impl();
    }

private:
    NullaryCallable impl;
};

} // namespace rmngr

