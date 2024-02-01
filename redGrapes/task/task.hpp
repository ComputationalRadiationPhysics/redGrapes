/* Copyright 2022 Michael Sippel
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <redGrapes/task/property/graph.hpp>
#include <redGrapes/task/property/id.hpp>
#include <redGrapes/task/property/inherit.hpp>
#include <redGrapes/task/property/queue.hpp>
#include <redGrapes/task/property/resource.hpp>
#include <redGrapes/task/property/trait.hpp>

#include <type_traits>

// defines REDGRAPES_TASK_PROPERTIES
#include <redGrapes_config.hpp>

namespace redGrapes
{

    using TaskProperties = TaskProperties1<
        GraphProperty,
        ResourceProperty
//  ,  QueueProperty
#ifdef REDGRAPES_TASK_PROPERTIES
        ,
        REDGRAPES_TASK_PROPERTIES
#endif
        ,
        IDProperty>;

    struct Task : TaskProperties
    {
        uint16_t arena_id;
        std::atomic<uint8_t> removal_countdown;

        Task() : removal_countdown(2)
        {
        }

        virtual ~Task()
        {
        }

        inline scheduler::EventPtr operator()()
        {
            return this->run();
        }

        virtual scheduler::EventPtr run() = 0;

        virtual void yield(scheduler::EventPtr event)
        {
            spdlog::error("Task {} does not support yield()", this->task_id);
        }

        virtual void* get_result_data()
        {
            return nullptr;
        }
    };

    template<typename Result>
    struct ResultTask : Task
    {
        Result result_data;

        virtual ~ResultTask()
        {
        }

        virtual void* get_result_data()
        {
            return &result_data;
        }

        virtual Result run_result() = 0;

        virtual scheduler::EventPtr run()
        {
            result_data = run_result();
            get_result_set_event().notify(); // result event now ready
            return scheduler::EventPtr{};
        }
    };

    template<>
    struct ResultTask<void> : Task
    {
        virtual ~ResultTask()
        {
        }

        virtual void run_result()
        {
        }

        virtual scheduler::EventPtr run()
        {
            run_result();
            get_result_set_event().notify();
            return scheduler::EventPtr{};
        }
    };

    template<typename F>
    struct FunTask : ResultTask<typename std::result_of<F()>::type>
    {
        std::optional<F> impl;

        virtual ~FunTask()
        {
        }

        typename std::result_of<F()>::type run_result()
        {
            return (*this->impl)();
        }
    };

} // namespace redGrapes

#include <redGrapes/scheduler/event.hpp>

#include <boost/context/continuation.hpp>

namespace redGrapes
{

    template<typename F>
    struct ContinuableTask : FunTask<F>
    {
        boost::context::continuation yield_cont;
        boost::context::continuation resume_cont;
        scheduler::EventPtr event;

        scheduler::EventPtr run()
        {
            if(!resume_cont)
            {
                resume_cont = boost::context::callcc(
                    [this](boost::context::continuation&& c)
                    {
                        this->yield_cont = std::move(c);
                        this->FunTask<F>::run();
                        this->event = scheduler::EventPtr{};

                        return std::move(this->yield_cont);
                    });
            }
            else
                resume_cont = resume_cont.resume();

            return event;
        }

        void yield(scheduler::EventPtr e)
        {
            this->event = e;
            yield_cont = yield_cont.resume();
        }
    };

} // namespace redGrapes
