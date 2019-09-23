
#pragma once

namespace rmngr
{

template < typename TaskProperties >
struct Task
{
    Task( TaskProperties const & properties )
        : properties( properties )
        , before_hook( []{} )
        , after_hook( []{} )
    {}

    virtual ~Task() = default;
    virtual void run() = 0;

    void operator() ()
    {
        before_hook();
        run();
        after_hook();
    }

    void hook_before( std::function<void(void)> const & hook )
    {
        before_hook = [rest=std::move(before_hook), hook]{ hook(); rest(); };
    }

    void hook_after( std::function<void(void)> const & hook )
    {
        after_hook = [rest=std::move(after_hook), hook]{ rest(); hook(); };
    }

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

