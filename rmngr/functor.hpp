#pragma once

#include <vector>
#include <string>
#include <boost/preprocessor/stringize.hpp>

#include <rmngr/queue.hpp>
#include <rmngr/resource.hpp>

namespace rmngr
{

class Functor
{
    public:
        struct CheckFunctor
        {
            static inline bool check(Functor const& a, Functor const& b)
            {
                for(ResourceAccess const& ra : a.resource_list)
                {
                    for(ResourceAccess const& rb : b.resource_list)
                    {
                        if(check_dependency(ra, rb))
                            return true;
                    }
                }
                return false;
            }
        };

        struct Label
        {
            static inline std::string getLabel(Functor const& f)
            {
                std::string label;
                label.append(f.name);
                label.append("\n");
                for(ResourceAccess const& a : f.resource_list)
                {
                    label.append(std::to_string(a.resourceID));
                    if(a.write)
                        label.append("w");
                    else
                        label.append("r");
                }

                return label;
            }
        };

        Functor(Queue<Functor, CheckFunctor, Label>& queue_, std::string const& name_, std::vector<ResourceAccess> const& ral)
            : queue(queue_), name(name_), resource_list(ral)
        {
        }

        std::string name;

        void operator() (void)
        {
            this->queue.push(*this);
        }

    private:
        std::vector<ResourceAccess> resource_list;
        Queue<Functor, CheckFunctor, Label>& queue;
};

#define FUNCTOR(name, queue, ...) \
    Functor name (queue, BOOST_PP_STRINGIZE(name), { __VA_ARGS__ });

} // namespace rmngr

