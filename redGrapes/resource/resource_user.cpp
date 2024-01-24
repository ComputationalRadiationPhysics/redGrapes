
#include <redGrapes/redGrapes.hpp>
#include <redGrapes/resource/resource.hpp>
#include <redGrapes/resource/resource_user.hpp>
#include <redGrapes/task/task.hpp>
#include <redGrapes/util/trace.hpp>

namespace redGrapes
{
    bool ResourceUsageEntry::operator==(ResourceUsageEntry const& other) const
    {
        return resource == other.resource;
    }

    ResourceUser::ResourceUser()
        : scope_level(SingletonContext::get().scope_depth())
        , access_list(memory::Allocator())
        , unique_resources(memory::Allocator())
    {
    }

    ResourceUser::ResourceUser(ResourceUser const& other)
        : scope_level(other.scope_level)
        , access_list(memory::Allocator(), other.access_list)
        , unique_resources(memory::Allocator(), other.unique_resources)
    {
    }

    ResourceUser::ResourceUser(std::initializer_list<ResourceAccess> list)
        : scope_level(scope_depth())
        , access_list(memory::Allocator())
        , unique_resources(memory::Allocator())
    {
        for(auto& ra : list)
            add_resource_access(ra);
    }

    void ResourceUser::add_resource_access(ResourceAccess ra)
    {
        this->access_list.push(ra);
        std::shared_ptr<ResourceBase> r = ra.get_resource();
        // unique_resources.erase(ResourceEntry{ r, r->users.end() });
        unique_resources.push(ResourceUsageEntry{r, r->users.rend()});
    }

    void ResourceUser::rm_resource_access(ResourceAccess ra)
    {
        this->access_list.erase(ra);
    }

    void ResourceUser::build_unique_resource_list()
    {
        for(auto ra = access_list.rbegin(); ra != access_list.rend(); ++ra)
        {
            std::shared_ptr<ResourceBase> r = ra->get_resource();
            unique_resources.erase(ResourceUsageEntry{r, r->users.rend()});
            unique_resources.push(ResourceUsageEntry{r, r->users.rend()});
        }
    }

    bool ResourceUser::has_sync_access(std::shared_ptr<ResourceBase> res)
    {
        for(auto ra = access_list.rbegin(); ra != access_list.rend(); ++ra)
        {
            if(ra->get_resource() == res && ra->is_synchronizing())
                return true;
        }
        return false;
    }

    bool ResourceUser::is_serial(ResourceUser const& a, ResourceUser const& b)
    {
        TRACE_EVENT("ResourceUser", "is_serial");
        for(auto ra = a.access_list.crbegin(); ra != a.access_list.crend(); ++ra)
            for(auto rb = b.access_list.crbegin(); rb != b.access_list.crend(); ++rb)
            {
                TRACE_EVENT("ResourceUser", "RA::is_serial");
                if(ResourceAccess::is_serial(*ra, *rb))
                    return true;
            }
        return false;
    }

    bool ResourceUser::is_superset_of(ResourceUser const& a) const
    {
        TRACE_EVENT("ResourceUser", "is_superset");
        for(auto ra = a.access_list.rbegin(); ra != a.access_list.rend(); ++ra)
        {
            bool found = false;
            for(auto r = access_list.rbegin(); r != access_list.rend(); ++r)
                if(r->is_superset_of(*ra))
                    found = true;

            if(!found && ra->scope_level() <= scope_level)
                // a introduced a new resource
                return false;
        }
        return true;
    }

    bool ResourceUser::is_superset(ResourceUser const& a, ResourceUser const& b)
    {
        return a.is_superset_of(b);
    }

} // namespace redGrapes
