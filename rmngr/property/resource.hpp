
/**
 * @file rmngr/property/resource.hpp
 */

#pragma once

#include <stdexcept>
#include <cstdarg>
#include <rmngr/resource/resource_user.hpp>

namespace rmngr
{

struct ResourceProperty : ResourceUser
{
    template < typename PropertiesBuilder >
    struct Builder
    {
        PropertiesBuilder & builder;
        Builder( PropertiesBuilder & b )
            : builder( b )
        {}

        PropertiesBuilder resources( std::initializer_list<ResourceAccess> list )
        {
	    builder.prop.access_list = list;
            return builder;
        }
    };

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

struct ResourceEnqueuePolicy
{
    static bool is_serial(ResourceProperty const & a, ResourceProperty const & b)
    {
        return rmngr::ResourceUser::is_serial( a, b );
    }
    static void assert_superset(ResourceProperty const & super, ResourceProperty const & sub)
    {
        if(! rmngr::ResourceUser::is_superset( super, sub ))
        {
            std::stringstream stream;
            stream << "Not allowed: " << std::endl
		   << super << std::endl
		   << "is no superset of " << std::endl
	           << sub << std::endl;

            throw std::runtime_error(stream.str());
        }
    }
};

} // namespace rmngr

