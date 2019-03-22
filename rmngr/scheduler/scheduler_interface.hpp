
/**
 * @file rmngr/scheduler/scheduler_interface.hpp
 */

#pragma once

#include <mutex>

namespace rmngr
{

struct SchedulerInterface
{
    struct SchedulableInterface
        : virtual public DelayedFunctorInterface
    {
        virtual ~SchedulableInterface()
        {
        }

        virtual void
        start( void ) = 0;

        virtual void
        finish( void ) = 0;
    };

    class WorkerInterface
    {
    protected:
        virtual void
        work( void ) = 0;

    public:
        void
        operator()( void )
        {
            this->work();
        }
    };

    virtual void
    update( void ) = 0;

    virtual bool
    empty( void ) = 0;

    virtual size_t
    num_threads( void ) const = 0;

    std::unique_lock< std::mutex >
    lock( void )
    {
        return std::unique_lock< std::mutex >( this->mutex );
    };

    void
    set_worker( WorkerInterface & worker )
    {
        this->worker = &worker;
    }

    template< typename Policy >
    struct ProtoProperty
    {
        typename Policy::ProtoProperty prop;
        operator typename Policy::ProtoProperty & ()
        {
            return this->prop;
        }
    };

    template< typename Policy >
    struct RuntimeProperty
    {
        typename Policy::RuntimeProperty prop;
        operator typename Policy::RuntimeProperty & ()
        {
            return this->prop;
        }
    };

    template< typename Policy >
    static typename Policy::ProtoProperty &
    proto_property( ProtoProperty< Policy > & s )
    {
        return s.prop;
    }

    template< typename Policy >
    static typename Policy::RuntimeProperty &
    runtime_property( RuntimeProperty< Policy > & s )
    {
        return s.prop;
    }

protected:
    std::mutex mutex;
    WorkerInterface* worker;
};

} // namespace rmngr

