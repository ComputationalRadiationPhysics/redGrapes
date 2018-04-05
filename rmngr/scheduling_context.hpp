
/**
 * @file rmngr/scheduling_context.hpp
 */

#pragma once

#include <boost/graph/adjacency_list.hpp>

#include <mutex>
#include <utility>
#include <fstream>
#include <rmngr/resource_user.hpp>
#include <rmngr/functor.hpp>
#include <rmngr/functor_queue.hpp>
#include <rmngr/precedence_graph.hpp>
#include <rmngr/scheduling_graph.hpp>
#include <rmngr/thread_dispatcher.hpp>
#include <rmngr/observer_ptr.hpp>

namespace rmngr
{

/** Manages scheduling-policies and the transition to dispatching the jobs.
 */
template<
    std::size_t n_threads=1
    //typename DispatchPolicy = FIFO,
    //... /* scheduling policies (e.g. resource user, label, main_thread, exclusive,..) */
>
class SchedulingContext
{
    public:
        /**
         * Base class storing all scheduling info and the functor
         */
        class Schedulable : public ResourceUser, virtual public DelayedFunctorInterface
        /*
          : boost::mpl::inherit_linearly
          <
          boost::mpl::set<SchedulingPolicies...>,
          boost::mpl::inherit< boost::mpl::_1, scheduling_traits<boost::mpl::_2>::flags >
          >::type
         */
        {
            public:
                Schedulable(std::vector<std::shared_ptr<ResourceAccess>> const& access_list_, std::string label_= {})
                    : ResourceUser(access_list_), state(pending), label(label_)
                {}

                enum { pending, ready, running, done, } state;
                std::string label;
        }; // class Schedulable

        template <typename DelayedFunctor>
        class SchedulableFunctor : public DelayedFunctor, public Schedulable
        {
            public:
                SchedulableFunctor(DelayedFunctor&& f, std::vector<std::shared_ptr<ResourceAccess>> const& access_list_, std::string label_)
                    : DelayedFunctor(std::forward<DelayedFunctor>(f)), Schedulable(access_list_, label_) {}
        }; // class SchedulableFunctor

        template <typename Functor>
        class ProtoSchedulableFunctor : public ResourceUser, public Functor
        {
            public:
                ProtoSchedulableFunctor(Functor const& f, std::vector<std::shared_ptr<ResourceAccess>> const& resource_list= {}, std::string label_= {})
                    : ResourceUser(resource_list), Functor(f), label(label_) {}

                template <typename DelayedFunctor>
                SchedulableFunctor<DelayedFunctor>* clone(DelayedFunctor&& f) const
                {
                    return new SchedulableFunctor<DelayedFunctor>(std::forward<DelayedFunctor>(f), this->access_list, this->label);
                }

                std::string label;
        }; // class ProtoSchedulableFunctor

        struct ReadyMarker
        {
            SchedulingContext& context;
            void operator() (Schedulable* s)
            {
                if(s->state == Schedulable::pending)
                {
                    s->state = Schedulable::ready;
                    context.dispatcher.push({&context, s});
                }
                else if(s->state == Schedulable::done)
                {
                    context.finish(s);
                }
            }
        }; // struct ReadyMarker

        struct Executor
        {
            SchedulingContext* context;
            Schedulable* s;

            void operator() (void)
            {
                context->write_graphviz();
                s->state = Schedulable::running;
                s->run();
                s->state = Schedulable::done;
                context->write_graphviz();

                std::lock_guard<std::mutex> lock(context->scheduler_mutex);
                context->finish(s);
            };
        }; // struct Executor

    public:
        using Graph = boost::adjacency_list<
            boost::setS,
            boost::vecS,
            boost::bidirectionalS,
            observer_ptr<Schedulable>
        >;

        std::mutex scheduler_mutex;
        SchedulingGraph<ReadyMarker, Graph> scheduler;
        ThreadDispatcher<Executor, n_threads> dispatcher;

        void finish(Schedulable* s)
        {
            if(this->scheduler.finish(s))
                delete s;
        }

    public:
        SchedulingContext(observer_ptr<RefinedGraph<Graph>> main_refinement)
            : scheduler(
                main_refinement,
                ReadyMarker{*this}
            )
        {}

        void write_graphviz(void)
        {
            std::lock_guard<std::mutex> lock(this->scheduler_mutex);

            static int step = 0;
            ++step;
            std::string name = std::string("Step ") + std::to_string(step);
            std::string path = std::string("step_") + std::to_string(step) + std::string(".dot");
            std::cout << "write schedulinggraph to " << path << std::endl;
            std::ofstream file(path);
            this->scheduler.write_graphviz(
                file,
                boost::make_function_property_map<Schedulable*>([](Schedulable* const& s)
                {
                    return s->label;
                }),
                boost::make_function_property_map<Schedulable*>([](Schedulable* const& s)
                {
                    switch(s->state)
                    {
                        case Schedulable::done:
                            return std::string("grey");
                        case Schedulable::running:
                            return std::string("green");
                        case Schedulable::ready:
                            return std::string("yellow");
                        default:
                            return std::string("red");
                    }
                }),
                name
            );

            file.close();
        }
}; // class SchedulingContext

} // namespace rmngr

