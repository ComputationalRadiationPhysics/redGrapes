#pragma once

#include <vector>
#include <memory>
#include <boost/type_index.hpp>

#include <rmngr/dependency_manager.hpp>

namespace rmngr
{

class ResourceAccess
{
    public:
        virtual ~ResourceAccess() {}

        bool check_dependency(ResourceAccess const& r) const
        {
            if(r.type == this->type)
                return this->_check_dependency(r);
            else
                return false;
        }
    protected:
        boost::typeindex::type_index type;
        virtual bool _check_dependency(ResourceAccess const& r) const = 0;
};

template <typename AccessProperty>
class ResourceBase
{
    public:
        ResourceBase()
        {}

        class ThisResourceAccess : public ResourceAccess
        {
            public:
                ThisResourceAccess(ResourceBase<AccessProperty> const* const& resource_, std::vector<AccessProperty> const& prop_)
                    : resource(resource_), prop(prop_)
                {}

                ~ThisResourceAccess()
                {}

            private:
                bool _check_dependency(ResourceAccess const& a_) const
                {
                    ThisResourceAccess const& a = *static_cast<ThisResourceAccess const*>(&a_); // no dynamic cast needed, type checked in ResourceAccess
                    if(this->resource == a.resource)
                    {
                        for(auto p1 : this->prop)
                        {
                            for(auto p2 : a.prop)
                            {
                                if(this->resource->check_dependency(p1, p2))
                                    return true;
                            }
                        }
                    }

                    return false;
                }

                ResourceBase<AccessProperty> const* resource;
                std::vector<AccessProperty> prop;
        }; // struct ResourceAccess

        std::shared_ptr<ThisResourceAccess> make_access(std::vector<AccessProperty> const& prop) const
        {
            return std::make_shared<ThisResourceAccess>(this, prop);
        }

    private:
        virtual bool check_dependency(AccessProperty const& a, AccessProperty const& b) const
        {
            return true;
        }
}; // class ResourceBase

template <typename AccessType, typename Dependency=BoolDependency>
class StaticResource : public ResourceBase<typename AccessType::Id>
{
    public:
        using Id = typename AccessType::Id;

        StaticResource()
        {
            if(! init)
            {
                AccessType::build_dependencies(access_dep);
                init = true;
            }
        }

        static DependencyManager<Id, Dependency> access_dep;
        static bool init;

    private:
        bool check_dependency(Id const& a, Id const& b) const
        {
            return access_dep.check_dependency(a, b);
        }
}; // class StaticResource

template<typename AccessType, typename Dependency>
DependencyManager<typename AccessType::Id, Dependency> StaticResource<AccessType, Dependency>::access_dep;
template<typename AccessType, typename Dependency>
bool StaticResource<AccessType, Dependency>::init = false;

} // namespace rmngr

