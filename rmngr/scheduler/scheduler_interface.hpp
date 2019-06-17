
/**
 * @file rmngr/scheduler/scheduler_interface.hpp
 */

#pragma once

#include <mutex>
#include <functional>

namespace rmngr
{

struct SchedulerInterface
{
    struct TaskInterface
    {
        virtual ~TaskInterface() = default;
        virtual void run( void ) = 0;
        virtual void finish( void ) = 0;
    };

    class WorkerInterface
    {
    protected:
        virtual void
        work( std::function<bool()> const& ) = 0;

    public:
        template <typename Pred>
        void
        operator() ( Pred const& pred )
        {
            this->work( std::function<bool()>(pred) );
        }
    };

    virtual void
    update( void ) = 0;

    virtual bool
    empty( void ) = 0;

    virtual size_t
    num_threads( void ) const = 0;

    std::unique_lock< std::recursive_mutex >
    lock( void )
    {
        return std::unique_lock< std::recursive_mutex >( this->mutex );
    };

    void
    set_worker( WorkerInterface & worker )
    {
        this->worker = &worker;
    }

    template< typename Policy >
    struct Property
    {
        using Patch = typename Policy::Property::Patch;
        typename Policy::Property prop;
        operator typename Policy::Property & ()
        {
            return this->prop;
        }
    };

    template< typename Policy >
    struct PropertyPatch
    {
        typename Policy::Property::Patch patch;
        operator typename Policy::Property::Patch & ()
        {
            return this->patch;
        }

        operator typename Policy::Property::Patch const & () const
        {
            return this->patch;
        }
    };

    template< typename Policy >
    static typename Policy::Property &
    property( Property< Policy > & s )
    {
        return s.prop;
    }

protected:
    std::recursive_mutex mutex;
    WorkerInterface* worker;
};

} // namespace rmngr

