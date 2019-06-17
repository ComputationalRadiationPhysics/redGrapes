
#pragma once

#include <stdexcept>
#include <rmngr/resource/resource_user.hpp>
#include <rmngr/scheduler/scheduler.hpp>

namespace rmngr
{

struct ResourceUserPolicy : DefaultSchedulingPolicy
{
    struct Property : ResourceUser
    {
        struct Patch
        {
            enum DiffType { ADD, REMOVE };
            std::list<std::pair<DiffType, ResourceAccess>> diff;

            void operator+= (Patch const& other)
            {
                this->diff.insert(std::end(this->diff), std::begin(other.diff), std::end(other.diff));
            }

            void operator+= (ResourceAccess const & ra)
            {
                this->diff.push_back(std::make_pair(DiffType::ADD, ra));
            }

            void operator-= (ResourceAccess const & ra)
            {
                this->diff.push_back(std::make_pair(DiffType::REMOVE, ra));
            }
        };

        void operator+= (ResourceAccess const & ra)
        {
            this->access_list.push_back(ra);
        }

        void operator-= (ResourceAccess const & ra)
        {
            this->access_list.remove(ra);
        }

        void apply_patch(Patch const & patch)
        {
            ResourceUser before = *this;

            for( auto x : patch.diff )
            {
                switch(x.first)
                {
                case Patch::DiffType::ADD:
                    (*this) += x.second;
                    break;
                case Patch::DiffType::REMOVE:
                    (*this) -= x.second;
                    break;
                }
            }

            if( ! before.is_superset_of(*this) )
                throw std::runtime_error("rmngr: ResourceUserPolicy: updated access list is no subset!");
        }
    };
};

template <typename T>
struct ResourceEnqueuePolicy
{
    static bool is_serial(T const & a, T const & b)
    {
        return rmngr::ResourceUser::is_serial(
                   a->template proto_property< rmngr::ResourceUserPolicy >(),
		   b->template proto_property< rmngr::ResourceUserPolicy >());
    }
    static void assert_superset(T const & super, T const & sub)
    {
        auto r_super = super->template proto_property< rmngr::ResourceUserPolicy >();
        auto r_sub = sub->template proto_property< rmngr::ResourceUserPolicy >();
        if(! rmngr::ResourceUser::is_superset( r_super, r_sub ))
        {
            std::stringstream stream;
            stream << "Not allowed: " << std::endl
		   << r_super << std::endl
		   << "is no superset of " << std::endl
	           << r_sub << std::endl;
            throw std::runtime_error(stream.str());
        }
    }
};

} // namespace rmngr

