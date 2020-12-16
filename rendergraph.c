#include "rendergraph.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef RENDERGRAPH_FEATURE_VULKAN

#define VK_NO_PROTOTYPES

#if defined(__linux__)
#define VK_USE_PLATFORM_XLIB_KHR
#define VK_USE_PLATFORM_WAYLAND_KHR

#elif defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#else
#error Unsupported OS
#endif

#define VOLK_IMPLEMENTATION
#include "volk.h"

#define RG_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define RG_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define RG_CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : (x) > (hi) ? (hi) : (x))
#define RG_LENGTH(array) (sizeof(array) / sizeof(array[0]))

enum {
    RG_FRAMES_IN_FLIGHT = 2,
    RG_MAX_SHADER_STAGES = 4,
    RG_MAX_COLOR_ATTACHMENTS = 16,
    RG_MAX_VERTEX_ATTRIBUTES = 8,
    RG_MAX_DESCRIPTOR_SETS = 8,
    RG_MAX_DESCRIPTOR_BINDINGS = 8,
    RG_MAX_DESCRIPTOR_TYPES = 8,
    RG_SETS_PER_PAGE = 32,
    RG_BUFFER_POOL_CHUNK_SIZE = 65536,
};

#define VK_CHECK(result)                                                                  \
    do                                                                                    \
    {                                                                                     \
        VkResult temp_result = result;                                                    \
        if (temp_result < 0)                                                              \
        {                                                                                 \
            fprintf(stderr, "%s:%u vulkan error: %d\n", __FILE__, __LINE__, temp_result); \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

// Hashing {{{
static void fnvHashReset(uint64_t *hash)
{
    *hash = 14695981039346656037ULL;
}

static void fnvHashUpdate(uint64_t *hash, uint8_t *bytes, size_t count)
{
    for (uint64_t i = 0; i < count; ++i)
    {
        *hash = ((*hash) * 1099511628211) ^ bytes[i];
    }
}
// }}}

// Array {{{
#define ARRAY_INITIAL_CAPACITY 16

static void *arrayGrow(void *ptr, size_t *cap, size_t wanted_cap, size_t item_size)
{
    if (!ptr)
    {
        size_t desired_cap = ((wanted_cap == 0) ? ARRAY_INITIAL_CAPACITY : wanted_cap);
        *cap = desired_cap;
        return malloc(item_size * desired_cap);
    }

    size_t desired_cap = ((wanted_cap == 0) ? ((*cap) * 2) : wanted_cap);
    *cap = desired_cap;
    return realloc(ptr, (desired_cap * item_size));
}

#define ARRAY_OF(type)                                                                   \
    struct                                                                               \
    {                                                                                    \
        type *ptr;                                                                       \
        size_t len;                                                                      \
        size_t cap;                                                                      \
    }

#define arrFull(a) ((a)->ptr ? ((a)->len >= (a)->cap) : 1)

#define arrPush(a, item)                                                                 \
    (arrFull(a) ? (a)->ptr = arrayGrow((a)->ptr, &(a)->cap, 0, sizeof(*((a)->ptr))) : 0, \
     (a)->ptr[(a)->len++] = (item))

#define arrPop(a) ((a)->len > 0 ? ((a)->len--, &(a)->ptr[(a)->len]) : NULL)

#define arrFree(a)                                                                       \
    do                                                                                   \
    {                                                                                    \
        (a)->ptr = NULL;                                                                 \
        (a)->len = 0;                                                                    \
        (a)->cap = 0;                                                                    \
    } while (0)
// }}}

// Hashmap {{{
typedef struct RgHashmap
{
    uint64_t size;
    uint64_t *hashes;
    uint64_t *values;
} RgHashmap;

static void rgHashmapInit(RgHashmap *hashmap, size_t size);
static void rgHashmapDestroy(RgHashmap *hashmap);
static void rgHashmapGrow(RgHashmap *hashmap);
static void rgHashmapSet(RgHashmap *hashmap, uint64_t hash, uint64_t value);
static uint64_t *rgHashmapGet(RgHashmap *hashmap, uint64_t hash);

static void rgHashmapInit(RgHashmap *hashmap, size_t size)
{
    memset(hashmap, 0, sizeof(*hashmap));

    hashmap->size = size;
    assert(hashmap->size > 0);

    // Round up to nearest power of two
    hashmap->size -= 1;
    hashmap->size |= hashmap->size >> 1;
    hashmap->size |= hashmap->size >> 2;
    hashmap->size |= hashmap->size >> 4;
    hashmap->size |= hashmap->size >> 8;
    hashmap->size |= hashmap->size >> 16;
    hashmap->size |= hashmap->size >> 32;
    hashmap->size += 1;

    // Init memory
    hashmap->hashes = (uint64_t *)malloc(hashmap->size * sizeof(uint64_t));
    memset(hashmap->hashes, 0, hashmap->size * sizeof(uint64_t));
    hashmap->values = (uint64_t *)malloc(hashmap->size * sizeof(uint64_t));
    memset(hashmap->values, 0, hashmap->size * sizeof(uint64_t));
}

static void rgHashmapDestroy(RgHashmap *hashmap)
{
    free(hashmap->values);
    free(hashmap->hashes);
}

static void rgHashmapGrow(RgHashmap *hashmap)
{
    uint64_t old_size = hashmap->size;
    uint64_t *old_hashes = hashmap->hashes;
    uint64_t *old_values = hashmap->values;

    hashmap->size *= 2;
    hashmap->hashes = (uint64_t *)malloc(hashmap->size * sizeof(uint64_t));
    memset(hashmap->hashes, 0, hashmap->size * sizeof(uint64_t));
    hashmap->values = (uint64_t *)malloc(hashmap->size * sizeof(uint64_t));
    memset(hashmap->values, 0, hashmap->size * sizeof(uint64_t));

    for (uint64_t i = 0; i < old_size; i++)
    {
        if (old_hashes[i] != 0)
        {
            rgHashmapSet(hashmap, old_hashes[i], old_values[i]);
        }
    }

    free(old_hashes);
    free(old_values);
}

static void rgHashmapSet(RgHashmap *hashmap, uint64_t hash, uint64_t value)
{
    assert(hash != 0);

    uint64_t i = hash & (hashmap->size - 1); // hash % size
    uint64_t iters = 0;

    while ((hashmap->hashes[i] != hash) && hashmap->hashes[i] != 0 &&
           iters < hashmap->size)
    {
        i = (i + 1) & (hashmap->size - 1); // (i+1) % size
        iters += 1;
    }

    if (iters >= hashmap->size)
    {
        rgHashmapGrow(hashmap);
        rgHashmapSet(hashmap, hash, value);
        return;
    }

    hashmap->hashes[i] = hash;
    hashmap->values[i] = value;
}

static uint64_t *rgHashmapGet(RgHashmap *hashmap, uint64_t hash)
{
    uint64_t i = hash & (hashmap->size - 1); // hash % size
    uint64_t iters = 0;

    while ((hashmap->hashes[i] != hash) && hashmap->hashes[i] != 0 &&
           iters < hashmap->size)
    {
        i = (i + 1) & (hashmap->size - 1); // (i+1) % size
        iters += 1;
    }

    if (iters >= hashmap->size)
    {
        return NULL;
    }

    if (hashmap->hashes[i] != 0)
    {
        return &hashmap->values[i];
    }

    return NULL;
}
// }}}

// Types {{{
typedef enum RgAllocationType
{
    RG_ALLOCATION_TYPE_UNKNOWN,
    RG_ALLOCATION_TYPE_GPU_ONLY,
    RG_ALLOCATION_TYPE_CPU_TO_GPU,
    RG_ALLOCATION_TYPE_GPU_TO_CPU,
} RgAllocationType;

typedef struct RgMemoryChunk
{
    size_t used;
    bool split;
} RgMemoryChunk;

typedef struct RgMemoryBlock
{
    VkDeviceMemory handle;
    size_t size;
    uint32_t memory_type_index;
    RgAllocationType type;

    RgMemoryChunk *chunks;
    uint32_t chunk_count;
    void *mapping;
} RgMemoryBlock;

typedef struct RgAllocator
{
    RgDevice *device;
    ARRAY_OF(RgMemoryBlock*) blocks;
} RgAllocator;

typedef struct RgAllocationInfo
{
    RgAllocationType type;
    VkMemoryRequirements requirements;
    bool dedicated;
} RgAllocationInfo;

typedef struct RgAllocation
{
    size_t size;
    size_t offset;
    bool dedicated;
    union
    {
        struct
        {
            RgMemoryBlock *block;
            size_t chunk_index;
        };
        struct
        {
            VkDeviceMemory dedicated_memory;
            void *dedicated_mapping;
        };
    };
} RgAllocation;

struct RgDevice
{
    RgDeviceInfo info;

    VkInstance instance;
    VkDebugUtilsMessengerEXT debug_callback;

    VkPhysicalDeviceProperties physical_device_properties;
    VkPhysicalDeviceFeatures physical_device_features;
    VkPhysicalDeviceMemoryProperties physical_device_memory_properties;

    VkQueueFamilyProperties *queue_family_properties;
    uint32_t num_queue_family_properties;

    VkPhysicalDevice physical_device;
    VkDevice device;

    RgAllocator *allocator;

    VkQueue graphics_queue;

    struct
    {
        uint32_t graphics;
        uint32_t compute;
    } queue_family_indices;
};

struct RgImage
{
    RgImageInfo info;
    VkImage image;
    VkImageView view;
    VkImageAspectFlags aspect;
    RgAllocation allocation;
};

struct RgBuffer
{
    RgBufferInfo info;
    VkBuffer buffer;
    RgAllocation allocation;
};

typedef struct RgBufferPool RgBufferPool;
typedef struct RgBufferChunk RgBufferChunk;

struct RgBufferChunk
{
    RgBufferPool *pool;
    RgBufferChunk *next;

    RgBuffer *buffer;
    size_t offset;
    size_t size;
    uint8_t *mapping;
};

struct RgBufferPool
{
    RgDevice *device;
    RgBufferChunk *base_chunk;
    size_t chunk_size;
    size_t alignment;
    RgBufferUsage usage;
};

typedef struct
{
    RgBuffer *buffer;
    uint8_t *mapping;
    size_t offset;
    size_t size;
} RgBufferAllocation;

typedef union
{
    VkDescriptorImageInfo image;
    VkDescriptorBufferInfo buffer;
} RgDescriptor;

typedef struct RgPass RgPass;
typedef struct RgNode RgNode;

struct RgCmdPool
{
    VkCommandPool command_pool;
};

struct RgCmdBuffer
{
    RgDevice *device;
    RgCmdPool *cmd_pool;
    RgPass *current_pass;
    VkCommandBuffer cmd_buffer;

    RgPipeline *current_pipeline;
    RgDescriptor bound_descriptors[RG_MAX_DESCRIPTOR_SETS][RG_MAX_DESCRIPTOR_BINDINGS];
    uint64_t set_hashes[RG_MAX_DESCRIPTOR_SETS];

    RgBufferPool ubo_pool;
    RgBufferPool vbo_pool;
    RgBufferPool ibo_pool;
};

typedef struct RgSwapchain
{
    RgDevice *device;

    RgFormat preferred_format;
    RgPlatformWindowInfo window;
    VkSurfaceKHR surface;
    VkSwapchainKHR swapchain;

    uint32_t present_family_index;
    VkQueue present_queue;

    uint32_t num_images;
    VkImage *images;
    VkImageView *image_views;

    VkFormat image_format;
    VkExtent2D extent;
    uint32_t current_image_index;
} RgSwapchain;

typedef struct RgGraphImageInfoInternal
{
    RgGraphImageScalingMode scaling_mode;
	float width;
	float height;
	uint32_t depth;
	uint32_t sample_count;
	uint32_t mip_count;
	uint32_t layer_count;
	RgFlags aspect;
	RgFlags usage;
	RgFormat format;
} RgGraphImageInfoInternal;

typedef enum RgResourceType
{
    RG_RESOURCE_IMAGE = 0,
    RG_RESOURCE_BUFFER = 1,
    RG_RESOURCE_EXTERNAL_IMAGE = 2,
    RG_RESOURCE_EXTERNAL_BUFFER = 3,
} RgResourceType;

typedef struct RgResource
{
    RgResourceType type;
    union
    {
        RgGraphImageInfoInternal image_info;
        RgBufferInfo buffer_info;
    };

    struct
    {
        union
        {
            RgImage *image;
            RgBuffer *buffer;
        };
    } frames[RG_FRAMES_IN_FLIGHT];
} RgResource;

typedef struct RgPassResource
{
    uint32_t index;
    RgResourceUsage pre_usage;
    RgResourceUsage post_usage;
} RgPassResource;

struct RgPass
{
    RgGraph *graph;
    uint32_t node_index;
    RgPassType type;

    bool is_backbuffer;

    VkRenderPass renderpass;
    VkExtent2D extent;
    uint64_t hash;
    uint32_t num_attachments;
    uint32_t num_color_attachments;
    bool has_depth_attachment;
    uint32_t depth_attachment_index;

    uint32_t num_framebuffers;
    struct
    {
        VkFramebuffer *framebuffers;
    } frames[RG_FRAMES_IN_FLIGHT];

    ARRAY_OF(RgPassResource) used_resources;

    VkClearValue *clear_values; // array with length equal to num_attachments

    VkFramebuffer current_framebuffer;
};

struct RgGraph
{
    RgDevice *device;
    RgCmdPool *cmd_pool;
    void *user_data;

    bool built;
    bool has_swapchain;
    RgSwapchain swapchain;

    uint32_t num_frames;

    ARRAY_OF(RgNode) nodes;
    ARRAY_OF(RgPass) passes;
    ARRAY_OF(RgResource) resources;

    VkSemaphore image_available_semaphores[RG_FRAMES_IN_FLIGHT];
    uint32_t current_frame;

    ARRAY_OF(VkBufferMemoryBarrier) buffer_barriers;
    ARRAY_OF(VkImageMemoryBarrier) image_barriers;
};

struct RgNode
{
    struct
    {
        RgCmdBuffer *cmd_buffer;
        VkSemaphore execution_finished_semaphore;

        ARRAY_OF(VkSemaphore) wait_semaphores;
        ARRAY_OF(VkPipelineStageFlags) wait_stages;

        VkFence fence;
    } frames[RG_FRAMES_IN_FLIGHT];

    uint32_t *pass_indices;
    uint32_t num_pass_indices;
};

typedef struct RgDescriptorPoolChunk
{
    struct RgDescriptorPoolChunk *next;
    VkDescriptorPool pool;
    VkDescriptorSet sets[RG_SETS_PER_PAGE];
    uint32_t allocated_count;
    RgHashmap map; // stores indices to allocated descriptor sets
} RgDescriptorPoolChunk;

typedef struct RgDescriptorPool
{
    RgDevice *device;
    RgDescriptorPoolChunk *base_chunk;
    VkDescriptorSetLayout set_layout;
    VkDescriptorUpdateTemplate update_template;
    VkDescriptorPoolSize pool_sizes[RG_MAX_DESCRIPTOR_TYPES];
    uint32_t num_pool_sizes;
    uint32_t num_bindings;
} RgDescriptorPool;

typedef enum RgPipelineType
{
    RG_PIPELINE_TYPE_GRAPHICS,
    RG_PIPELINE_TYPE_COMPUTE,
} RgPipelineType;

struct RgPipeline
{
    RgPipelineType type;

    uint32_t num_bindings;
    RgPipelineBinding *bindings;

    RgDescriptorPool pools[RG_MAX_DESCRIPTOR_SETS];
    uint32_t num_sets;
    VkPipelineLayout pipeline_layout;

    union
    {
        struct
        {
            RgHashmap instances;

            uint32_t vertex_stride;
            uint32_t num_vertex_attributes;
            RgVertexAttribute *vertex_attributes;

            RgPolygonMode       polygon_mode;
            RgCullMode          cull_mode;
            RgFrontFace         front_face;
            RgPrimitiveTopology topology;

            RgPipelineBlendState        blend;
            RgPipelineDepthStencilState depth_stencil;

            char* vertex_entry;
            char* fragment_entry;

            VkShaderModule vertex_shader;
            VkShaderModule fragment_shader;
        } graphics;

        struct
        {
            VkPipeline instance;
            VkShaderModule shader;
        } compute;
    };
};
// }}}

// Type conversions {{{
static VkFormat format_to_vk(RgFormat fmt)
{
    switch (fmt)
    {
    case RG_FORMAT_UNDEFINED: return VK_FORMAT_UNDEFINED;

    case RG_FORMAT_R8_UNORM: return VK_FORMAT_R8_UNORM;
    case RG_FORMAT_RG8_UNORM: return VK_FORMAT_R8G8_UNORM;
    case RG_FORMAT_RGB8_UNORM: return VK_FORMAT_R8G8B8_UNORM;
    case RG_FORMAT_RGBA8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;

    case RG_FORMAT_R8_UINT: return VK_FORMAT_R8_UINT;
    case RG_FORMAT_RG8_UINT: return VK_FORMAT_R8G8_UINT;
    case RG_FORMAT_RGB8_UINT: return VK_FORMAT_R8G8B8_UINT;
    case RG_FORMAT_RGBA8_UINT: return VK_FORMAT_R8G8B8A8_UINT;

    case RG_FORMAT_R16_UINT: return VK_FORMAT_R16_UINT;
    case RG_FORMAT_RG16_UINT: return VK_FORMAT_R16G16_UINT;
    case RG_FORMAT_RGB16_UINT: return VK_FORMAT_R16G16B16_UINT;
    case RG_FORMAT_RGBA16_UINT: return VK_FORMAT_R16G16B16A16_UINT;

    case RG_FORMAT_R32_UINT: return VK_FORMAT_R32_UINT;
    case RG_FORMAT_RG32_UINT: return VK_FORMAT_R32G32_UINT;
    case RG_FORMAT_RGB32_UINT: return VK_FORMAT_R32G32B32_UINT;
    case RG_FORMAT_RGBA32_UINT: return VK_FORMAT_R32G32B32A32_UINT;

    case RG_FORMAT_R32_SFLOAT: return VK_FORMAT_R32_SFLOAT;
    case RG_FORMAT_RG32_SFLOAT: return VK_FORMAT_R32G32_SFLOAT;
    case RG_FORMAT_RGB32_SFLOAT: return VK_FORMAT_R32G32B32_SFLOAT;
    case RG_FORMAT_RGBA32_SFLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;

    case RG_FORMAT_R16_SFLOAT: return VK_FORMAT_R16_SFLOAT;
    case RG_FORMAT_RG16_SFLOAT: return VK_FORMAT_R16G16_SFLOAT;
    case RG_FORMAT_RGBA16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;

    case RG_FORMAT_BGRA8_UNORM: return VK_FORMAT_B8G8R8A8_UNORM;
    case RG_FORMAT_BGRA8_SRGB: return VK_FORMAT_B8G8R8A8_SRGB;

    case RG_FORMAT_D32_SFLOAT_S8_UINT: return VK_FORMAT_D32_SFLOAT_S8_UINT;
    case RG_FORMAT_D32_SFLOAT: return VK_FORMAT_D32_SFLOAT;
    case RG_FORMAT_D24_UNORM_S8_UINT: return VK_FORMAT_D24_UNORM_S8_UINT;
    case RG_FORMAT_D16_UNORM_S8_UINT: return VK_FORMAT_D16_UNORM_S8_UINT;
    case RG_FORMAT_D16_UNORM: return VK_FORMAT_D16_UNORM;

    case RG_FORMAT_BC7_UNORM: return VK_FORMAT_BC7_UNORM_BLOCK;
    case RG_FORMAT_BC7_SRGB: return VK_FORMAT_BC7_SRGB_BLOCK;
    }
    assert(0);
    return 0;
}

static VkFilter filter_to_vk(RgFilter value)
{
    switch (value)
    {
    case RG_FILTER_LINEAR: return VK_FILTER_LINEAR;
    case RG_FILTER_NEAREST: return VK_FILTER_NEAREST;
    }
    assert(0);
    return 0;
}

static VkSamplerAddressMode address_mode_to_vk(RgSamplerAddressMode value)
{
    switch (value)
    {
    case RG_SAMPLER_ADDRESS_MODE_REPEAT: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case RG_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case RG_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE:
        return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    }
    assert(0);
    return 0;
}

static VkBorderColor border_color_to_vk(RgBorderColor value)
{
    switch (value)
    {
    case RG_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK:
        return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    case RG_BORDER_COLOR_INT_TRANSPARENT_BLACK:
        return VK_BORDER_COLOR_INT_TRANSPARENT_BLACK;
    case RG_BORDER_COLOR_FLOAT_OPAQUE_BLACK: return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    case RG_BORDER_COLOR_INT_OPAQUE_BLACK: return VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    case RG_BORDER_COLOR_FLOAT_OPAQUE_WHITE: return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    case RG_BORDER_COLOR_INT_OPAQUE_WHITE: return VK_BORDER_COLOR_INT_OPAQUE_WHITE;
    }
    assert(0);
    return 0;
}

static VkIndexType index_type_to_vk(RgIndexType index_type)
{
    switch (index_type)
    {
    case RG_INDEX_TYPE_UINT16: return VK_INDEX_TYPE_UINT16;
    case RG_INDEX_TYPE_UINT32: return VK_INDEX_TYPE_UINT32;
    }
    assert(0);
    return 0;
}

static VkCullModeFlagBits cull_mode_to_vk(RgCullMode cull_mode)
{
    switch (cull_mode)
    {
    case RG_CULL_MODE_NONE: return VK_CULL_MODE_NONE;
    case RG_CULL_MODE_BACK: return VK_CULL_MODE_BACK_BIT;
    case RG_CULL_MODE_FRONT: return VK_CULL_MODE_FRONT_BIT;
    case RG_CULL_MODE_FRONT_AND_BACK: return VK_CULL_MODE_FRONT_AND_BACK;
    }
    assert(0);
    return 0;
}

static VkFrontFace front_face_to_vk(RgFrontFace front_face)
{
    switch (front_face)
    {
    case RG_FRONT_FACE_CLOCKWISE: return VK_FRONT_FACE_CLOCKWISE;
    case RG_FRONT_FACE_COUNTER_CLOCKWISE: return VK_FRONT_FACE_COUNTER_CLOCKWISE;
    }
    assert(0);
    return 0;
}

static VkPolygonMode polygon_mode_to_vk(RgPolygonMode polygon_mode)
{
    switch (polygon_mode)
    {
    case RG_POLYGON_MODE_FILL: return VK_POLYGON_MODE_FILL;
    case RG_POLYGON_MODE_LINE: return VK_POLYGON_MODE_LINE;
    case RG_POLYGON_MODE_POINT: return VK_POLYGON_MODE_POINT;
    }
    assert(0);
    return 0;
}

static VkPrimitiveTopology primitive_topology_to_vk(RgPrimitiveTopology value)
{
    switch (value)
    {
    case RG_PRIMITIVE_TOPOLOGY_LINE_LIST: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case RG_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }
    assert(0);
    return 0;
}

static VkDescriptorType pipeline_binding_type_to_vk(RgPipelineBindingType type)
{
    switch (type)
    {
    case RG_BINDING_UNIFORM_BUFFER: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case RG_BINDING_STORAGE_BUFFER: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case RG_BINDING_IMAGE: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case RG_BINDING_SAMPLER: return VK_DESCRIPTOR_TYPE_SAMPLER;
    case RG_BINDING_IMAGE_SAMPLER: return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    assert(0);
    return 0;
}
// }}}

// Barrier utils {{{
static void rgResourceUsageToVk(
    RgResourceUsage usage,
    VkAccessFlags *access_flags,
    /* optional */ VkImageLayout *image_layout)
{
    switch (usage)
    {
    case RG_RESOURCE_USAGE_UNDEFINED:
    {
        if (access_flags)
            *access_flags = 0;
        if (image_layout)
            *image_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        break;
    }
    case RG_RESOURCE_USAGE_SAMPLED:
    {
        if (access_flags)
            *access_flags = VK_ACCESS_SHADER_READ_BIT;
        if (image_layout)
            *image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        break;
    }
    case RG_RESOURCE_USAGE_TRANSFER_SRC:
    {
        if (access_flags)
            *access_flags = VK_ACCESS_TRANSFER_READ_BIT;
        if (image_layout)
            *image_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        break;
    }
    case RG_RESOURCE_USAGE_TRANSFER_DST:
    {
        if (access_flags)
            *access_flags = VK_ACCESS_TRANSFER_WRITE_BIT;
        if (image_layout)
            *image_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        break;
    }
    case RG_RESOURCE_USAGE_COLOR_ATTACHMENT:
    {
        if (access_flags)
            *access_flags = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;

        // We use VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL because
        // this is the renderpass attachment's finalLayout
        if (image_layout)
            *image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        break;
    }
    case RG_RESOURCE_USAGE_DEPTH_STENCIL_ATTACHMENT:
    {
        if (access_flags)
            *access_flags = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        // We use VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL because
        // this is the renderpass attachment's finalLayout
        if (image_layout)
            *image_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        break;
    }
    }
}
// }}}

// Device memory allocator {{{
static int32_t
rgFindMemoryProperties2(
        const VkPhysicalDeviceMemoryProperties* memory_properties,
        uint32_t memory_type_bits_requirement,
        VkMemoryPropertyFlags required_properties)
{
    uint32_t memory_count = memory_properties->memoryTypeCount;

    for (uint32_t memory_index = 0; memory_index < memory_count; ++memory_index)
    {
        uint32_t memory_type_bits = (1 << memory_index);
        bool is_required_memory_type = memory_type_bits_requirement & memory_type_bits;

        VkMemoryPropertyFlags properties = memory_properties->memoryTypes[memory_index].propertyFlags;
        bool has_required_properties = (properties & required_properties) == required_properties;

        if (is_required_memory_type && has_required_properties)
        {
            return (int32_t)memory_index;
        }
    }

    // failed to find memory type
    return -1;
}

static VkResult rgFindMemoryProperties(
    RgAllocator *allocator,
    const RgAllocationInfo *info,
    int32_t *memory_type_index,
    VkMemoryPropertyFlagBits *required_properties)
{
    switch (info->type)
    {
    case RG_ALLOCATION_TYPE_GPU_ONLY:
        *required_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case RG_ALLOCATION_TYPE_CPU_TO_GPU:
        *required_properties =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        break;
    case RG_ALLOCATION_TYPE_GPU_TO_CPU:
        *required_properties =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        break;
    case RG_ALLOCATION_TYPE_UNKNOWN: break;
    }

    *memory_type_index = rgFindMemoryProperties2(
        &allocator->device->physical_device_memory_properties,
        info->requirements.memoryTypeBits,
        *required_properties);

    if (*memory_type_index == -1)
    {
        // We try again!

        switch (info->type)
        {
        case RG_ALLOCATION_TYPE_GPU_ONLY:
            *required_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            break;
        case RG_ALLOCATION_TYPE_CPU_TO_GPU:
            *required_properties =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
        case RG_ALLOCATION_TYPE_GPU_TO_CPU:
            *required_properties =
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            break;
        case RG_ALLOCATION_TYPE_UNKNOWN: break;
        }

        *memory_type_index = rgFindMemoryProperties2(
            &allocator->device->physical_device_memory_properties,
            info->requirements.memoryTypeBits,
            *required_properties);
    }

    if (*memory_type_index == -1)
    {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    return VK_SUCCESS;
}

static inline RgMemoryChunk *rgMemoryChunkParent(
    RgMemoryBlock *block,
    RgMemoryChunk *chunk)
{
    ptrdiff_t index = (ptrdiff_t)(chunk - block->chunks);
    assert(index >= 0);

    if (index == 0) return NULL;

    index = (index - 1) / 2;
    if (index < 0) return NULL;

    return &block->chunks[index];
}

static inline RgMemoryChunk *rgMemoryChunkLeftChild(
    RgMemoryBlock *block,
    RgMemoryChunk *chunk)
{
    ptrdiff_t index = (ptrdiff_t)(chunk - block->chunks);
    assert(index >= 0);

    index = 2 * index + 1;
    if (index >= block->chunk_count) return NULL;

    return &block->chunks[index];
}

static inline RgMemoryChunk *rgMemoryChunkRightChild(
    RgMemoryBlock *block,
    RgMemoryChunk *chunk)
{
    ptrdiff_t index = (ptrdiff_t)(chunk - block->chunks);
    assert(index >= 0);

    index = 2 * index + 2;
    if (index >= block->chunk_count) return NULL;

    return &block->chunks[index];
}

static inline size_t rgMemoryChunkSize(
    RgMemoryBlock *block,
    RgMemoryChunk *chunk)
{
    ptrdiff_t index = (ptrdiff_t)(chunk - block->chunks);
    assert(index >= 0);

    // Tree level of the chunk starting from 0
    size_t tree_level = floor(log2((double)(index+1)));

    // chunk_size = block->size / pow(2, tree_level);
    size_t chunk_size = block->size >> tree_level;
    assert(chunk_size >= 1);
    return chunk_size;
}

static inline size_t rgMemoryChunkOffset(
    RgMemoryBlock *block,
    RgMemoryChunk *chunk)
{
    ptrdiff_t index = (ptrdiff_t)(chunk - block->chunks);
    assert(index >= 0);

    if (index == 0) return 0;

    RgMemoryChunk *parent = rgMemoryChunkParent(block, chunk);
    assert(parent);

    size_t parent_offset = rgMemoryChunkOffset(block, parent);

    if (index & 1)
    {
        // Right
        parent_offset += rgMemoryChunkSize(block, chunk);
    }

    return parent_offset;
}

static inline void rgMemoryChunkUpdateUsage(
    RgMemoryBlock *block,
    RgMemoryChunk *chunk)
{
    if (chunk->split)
    {
        RgMemoryChunk *left = rgMemoryChunkLeftChild(block, chunk);
        RgMemoryChunk *right = rgMemoryChunkRightChild(block, chunk);
        chunk->used = left->used + right->used;
    }

    RgMemoryChunk *parent = rgMemoryChunkParent(block, chunk);
    if (parent)
    {
        rgMemoryChunkUpdateUsage(block, parent);
    }
}

static inline RgMemoryChunk *rgMemoryChunkSplit(
    RgMemoryBlock *block,
    RgMemoryChunk *chunk,
    size_t size,
    size_t alignment)
{
    assert(chunk);

    ptrdiff_t index = (ptrdiff_t)(chunk - block->chunks);
    assert(index >= 0);

    const size_t chunk_size = rgMemoryChunkSize(block, chunk);
    const size_t chunk_offset = rgMemoryChunkOffset(block, chunk);

    if ((chunk_size - chunk->used) < size) return NULL;

    RgMemoryChunk *left = rgMemoryChunkLeftChild(block, chunk);
    RgMemoryChunk *right = rgMemoryChunkRightChild(block, chunk);

    const size_t left_offset = chunk_offset;
    const size_t right_offset = chunk_offset + (chunk_size / 2);

    bool can_split = true;

    // We have to split into aligned chunks
    can_split &= ((left_offset % alignment == 0) || (right_offset % alignment == 0));

    // We have to be able to split into the required size
    can_split &= (size <= (chunk_size / 2));

    // We have to be able to have enough space to split
    can_split &= (chunk->used <= (chunk_size / 2));

    // chunk needs to have children in order to split
    can_split &= ((left != NULL) && (right != NULL));

    /* printf("index = %ld\n", index); */
    /* printf("size = %zu\n", size); */
    /* printf("alignment = %zu\n", alignment); */
    /* printf("chunk->split = %u\n", (uint32_t)chunk->split); */
    /* printf("chunk->used = %zu\n", chunk->used); */
    /* printf("chunk_size = %zu\n", chunk_size); */
    /* printf("chunk_offset = %zu\n", chunk_offset); */

    if (can_split)
    {
        if (!chunk->split)
        {
            // Chunk is not yet split, so do it now
            chunk->split = true;

            left->split = false;
            left->used = chunk->used;

            right->split = false;
            right->used = 0;
        }

        RgMemoryChunk *returned_chunk = NULL;

        returned_chunk = rgMemoryChunkSplit(block, left, size, alignment);
        if (returned_chunk) return returned_chunk;

        returned_chunk = rgMemoryChunkSplit(block, right, size, alignment);
        if (returned_chunk) return returned_chunk;
    }

    // We can't split, but if the chunk meets the requirements, return it
    if ((!chunk->split) &&
        (chunk->used == 0) &&
        (chunk_size >= size) &&
        (chunk_offset % alignment == 0))
    {
        return chunk;
    }

    return NULL;
}

static inline void rgMemoryChunkJoin(RgMemoryBlock *block, RgMemoryChunk *chunk)
{
    assert(chunk->split);

    RgMemoryChunk *left = rgMemoryChunkLeftChild(block, chunk);
    RgMemoryChunk *right = rgMemoryChunkRightChild(block, chunk);

    bool can_join = true;
    can_join &= (left->used == 0);
    can_join &= (!left->split);
    can_join &= (right->used == 0);
    can_join &= (!right->split);
    
    if (can_join)
    {
        // Join
        chunk->split = false;
        chunk->used = 0;

        RgMemoryChunk *parent = rgMemoryChunkParent(block, chunk);
        if (parent)
        {
            rgMemoryChunkJoin(block, parent);
        }
    }
}

static VkResult rgMemoryBlockAllocate(
    RgMemoryBlock *block,
    size_t size,
    size_t alignment,
    RgAllocation *allocation)
{
    assert(block->chunk_count > 0);
    RgMemoryChunk *chunk = &block->chunks[0];
    chunk = rgMemoryChunkSplit(
        block,
        chunk,
        size,
        alignment);

    if (!chunk)
    {
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }

    assert(chunk->used == 0);
    chunk->used = size;

    rgMemoryChunkUpdateUsage(block, chunk);

    size_t offset = rgMemoryChunkOffset(block, chunk);

    allocation->block = block;
    allocation->size = size;
    allocation->offset = offset;
    allocation->chunk_index = (size_t)(chunk - block->chunks);

    return VK_SUCCESS;
}

static void rgMemoryBlockFree(
    RgMemoryBlock *block,
    const RgAllocation *allocation)
{
    size_t chunk_index = allocation->chunk_index;
    RgMemoryChunk *chunk = &block->chunks[chunk_index];

    chunk->used = 0;
    rgMemoryChunkUpdateUsage(block, chunk);

    RgMemoryChunk *parent = rgMemoryChunkParent(block, chunk);
    if (parent) rgMemoryChunkJoin(block, parent);
}

static VkResult
rgAllocatorCreateMemoryBlock(
    RgAllocator *allocator,
    RgAllocationInfo *info,
    RgMemoryBlock **out_block)
{
    VkResult result = VK_SUCCESS;

    int32_t memory_type_index = -1;
    VkMemoryPropertyFlagBits required_properties = 0;
    result = rgFindMemoryProperties(allocator, info, &memory_type_index, &required_properties);
    if (result != VK_SUCCESS) return result;

    assert(memory_type_index >= 0);

    const uint64_t DEFAULT_DEVICE_MEMBLOCK_SIZE = 256 * 1024 * 1024;
    const uint64_t DEFAULT_HOST_MEMBLOCK_SIZE = 64 * 1024 * 1024;

    uint64_t memblock_size = 0;
    if (required_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    {
        memblock_size = DEFAULT_HOST_MEMBLOCK_SIZE;
    }
    else
    {
        memblock_size = DEFAULT_DEVICE_MEMBLOCK_SIZE;
    }

    memblock_size = RG_MAX(info->requirements.size, memblock_size);

    // Round memblock_size to the next power of 2
    memblock_size--;
    memblock_size |= memblock_size >> 1;
    memblock_size |= memblock_size >> 2;
    memblock_size |= memblock_size >> 4;
    memblock_size |= memblock_size >> 8;
    memblock_size |= memblock_size >> 16;
    memblock_size |= memblock_size >> 32;
    memblock_size++;

    assert(memblock_size >= info->requirements.size);

    VkMemoryAllocateInfo vk_allocate_info;
    memset(&vk_allocate_info, 0, sizeof(vk_allocate_info));
    vk_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vk_allocate_info.allocationSize = memblock_size;
    vk_allocate_info.memoryTypeIndex = (uint32_t)memory_type_index;

    VkDeviceMemory vk_memory = VK_NULL_HANDLE;
    result = vkAllocateMemory(
        allocator->device->device,
        &vk_allocate_info,
        NULL,
        &vk_memory);

    if (result != VK_SUCCESS) return result;

    void *mapping = NULL;

    if (required_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    {
        result = vkMapMemory(
            allocator->device->device,
            vk_memory,
            0,
            VK_WHOLE_SIZE,
            0,
            &mapping);

        if (result != VK_SUCCESS)
        {
            vkFreeMemory(allocator->device->device, vk_memory, NULL);
            return result;
        }
    }

    RgMemoryBlock *block = malloc(sizeof(*block));
    memset(block, 0, sizeof(*block));

    block->handle = vk_memory;
    block->mapping = mapping;
    block->size = memblock_size;
    block->memory_type_index = (uint32_t)memory_type_index;
    block->type = info->type;
    block->chunk_count = RG_MIN(2 * 256 - 1, 2 * memblock_size - 1);
    block->chunks = malloc(sizeof(*block->chunks) * block->chunk_count);
    memset(block->chunks, 0, sizeof(*block->chunks) * block->chunk_count);

    arrPush(&allocator->blocks, block);

    *out_block = block;
    
    return result;
}

static void
rgAllocatorFreeMemoryBlock(RgAllocator *allocator, RgMemoryBlock *block)
{
    if (block->mapping)
    {
        vkUnmapMemory(
            allocator->device->device,
            block->handle);
    }
    vkFreeMemory(allocator->device->device, block->handle, NULL);
    free(block->chunks);
    free(block);
}

static RgAllocator *rgAllocatorCreate(RgDevice *device)
{
    RgAllocator *allocator = malloc(sizeof(*allocator));
    memset(allocator, 0, sizeof(*allocator));

    allocator->device = device;

    return allocator;
}

static void rgAllocatorDestroy(RgAllocator *allocator)
{
    for (uint32_t i = 0; i < allocator->blocks.len; ++i)
    {
        RgMemoryBlock *block = allocator->blocks.ptr[i];
        rgAllocatorFreeMemoryBlock(allocator, block);
    }
    arrFree(&allocator->blocks);
    free(allocator);
}

static VkResult rgAllocatorAllocate(
    RgAllocator *allocator,
    RgAllocationInfo *info,
    RgAllocation *allocation)
{
    memset(allocation, 0, sizeof(*allocation));

    if (info->dedicated)
    {
        VkResult result = VK_SUCCESS;

        int32_t memory_type_index = -1;
        VkMemoryPropertyFlagBits required_properties = 0;
        result = rgFindMemoryProperties(allocator, info, &memory_type_index, &required_properties);
        if (result != VK_SUCCESS) return result;

        assert(memory_type_index >= 0);

        VkMemoryAllocateInfo vk_allocate_info;
        memset(&vk_allocate_info, 0, sizeof(vk_allocate_info));
        vk_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        vk_allocate_info.allocationSize = info->requirements.size;
        vk_allocate_info.memoryTypeIndex = (uint32_t)memory_type_index;

        VkDeviceMemory vk_memory = VK_NULL_HANDLE;
        result = vkAllocateMemory(
            allocator->device->device,
            &vk_allocate_info,
            NULL,
            &vk_memory);

        if (result != VK_SUCCESS) return result;

        void *mapping = NULL;

        if (required_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
        {
            result = vkMapMemory(
                allocator->device->device,
                vk_memory,
                0,
                VK_WHOLE_SIZE,
                0,
                &mapping);

            if (result != VK_SUCCESS)
            {
                vkFreeMemory(allocator->device->device, vk_memory, NULL);
                return result;
            }
        }

        allocation->dedicated = true;
        allocation->dedicated_memory = vk_memory;
        allocation->dedicated_mapping = mapping;
        allocation->size = info->requirements.size;
        allocation->offset = 0;

        return result;
    }

    for (int32_t i = ((int32_t)allocator->blocks.len)-1; i >= 0; --i)
    {
        RgMemoryBlock *block = allocator->blocks.ptr[i];

        if (info->type == block->type)
        {
            VkResult result = rgMemoryBlockAllocate(
                block,
                info->requirements.size,
                info->requirements.alignment,
                allocation);

            if (info->type == RG_ALLOCATION_TYPE_CPU_TO_GPU
                || info->type == RG_ALLOCATION_TYPE_GPU_TO_CPU)
            {
                assert(block->mapping);
            }

            if (result != VK_SUCCESS) continue;

            return VK_SUCCESS;
        }
    }

    // We could not allocate from any existing memory block, so let's make a new one
    RgMemoryBlock *block = NULL;
    VkResult result = rgAllocatorCreateMemoryBlock(allocator, info, &block);
    if (result != VK_SUCCESS) return result;

    result = rgMemoryBlockAllocate(
        block,
        info->requirements.size,
        info->requirements.alignment,
        allocation);

    return result;
}

static void rgAllocatorFree(RgAllocator *allocator, const RgAllocation *allocation)
{
    if (allocation->dedicated)
    {
        VK_CHECK(vkDeviceWaitIdle(allocator->device->device));
        if (allocation->dedicated_mapping)
        {
            vkUnmapMemory(
                allocator->device->device,
                allocation->dedicated_memory);
        }
        vkFreeMemory(allocator->device->device, allocation->dedicated_memory, NULL);
    }
    else
    {
        rgMemoryBlockFree(allocation->block, allocation);
    }
}

static VkResult rgMapAllocation(RgAllocator *allocator, const RgAllocation *allocation, void **ppData)
{
    (void)allocator;

    if (allocation->dedicated)
    {
        if (allocation->dedicated_mapping)
        {
            *ppData = allocation->dedicated_mapping;
            return VK_SUCCESS;
        }

        return VK_ERROR_MEMORY_MAP_FAILED;
    }

    assert(allocation->block->mapping);
    if (allocation->block->mapping)
    {
        *ppData = ((uint8_t*)allocation->block->mapping) + allocation->offset;
        return VK_SUCCESS;
    }
    return VK_ERROR_MEMORY_MAP_FAILED;
}

static void rgUnmapAllocation(RgAllocator *allocator, const RgAllocation *allocation)
{
    // No-op for now
    (void)allocator;
    (void)allocation;
}
// }}}

// Device {{{
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_message_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageTypes,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData)
{
    (void)messageSeverity;
    (void)messageTypes;
    (void)pUserData;
    fprintf(stderr, "Validation layer: %s\n", pCallbackData->pMessage);

    return VK_FALSE;
}

static bool check_layer_support(
    const char** required_layers,
    uint32_t required_layer_count)
{
    if (required_layer_count == 0) return true;

    uint32_t count;
    vkEnumerateInstanceLayerProperties(&count, NULL);
    VkLayerProperties *available_layers =
        (VkLayerProperties *)malloc(sizeof(VkLayerProperties) * count);
    vkEnumerateInstanceLayerProperties(&count, available_layers);

    for (uint32_t i = 0; i < required_layer_count; ++i)
    {
        const char *required_layer_name = required_layers[i];
        bool layer_found = false;

        for (uint32_t j = 0; j < count; ++j)
        {
            VkLayerProperties *layer = &available_layers[j];
            if (strcmp(layer->layerName, required_layer_name) == 0)
            {
                layer_found = true;
                break;
            }
        }

        if (!layer_found)
        {
            free(available_layers);
            return false;
        }
    }

    free(available_layers);
    return true;
}

static uint32_t get_queue_family_index(RgDevice *device, VkQueueFlagBits queue_flags)
{
    // Dedicated queue for compute
    // Try to find a queue family index that supports compute but not graphics
    if (queue_flags & VK_QUEUE_COMPUTE_BIT)
    {
        for (uint32_t i = 0; i < device->num_queue_family_properties; i++)
        {
            if ((device->queue_family_properties[i].queueFlags & queue_flags) &&
                ((device->queue_family_properties[i].queueFlags &
                  VK_QUEUE_GRAPHICS_BIT) == 0))
            {
                return i;
            }
        }
    }

    // For other queue types or if no separate compute queue is present,
    // return the first one to support the requested flags
    for (uint32_t i = 0; i < device->num_queue_family_properties; i++)
    {
        if (device->queue_family_properties[i].queueFlags & queue_flags)
        {
            return i;
        }
    }

    return UINT32_MAX;
}

RgDevice *rgDeviceCreate(RgDeviceInfo *info)
{
    RgDevice *device = (RgDevice *)malloc(sizeof(RgDevice));
    memset(device, 0, sizeof(*device));

    device->info = *info;

    VK_CHECK(volkInitialize());

    ARRAY_OF(const char*) layers;
    memset(&layers, 0, sizeof(layers));

    ARRAY_OF(const char*) instance_extensions;
    memset(&instance_extensions, 0, sizeof(instance_extensions));

    if (device->info.enable_validation)
    {
        fprintf(stderr, "Using validation layers\n");
        arrPush(&layers, "VK_LAYER_KHRONOS_validation");
        arrPush(&instance_extensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    if (!check_layer_support(layers.ptr, layers.len))
    {
        fprintf(stderr, "Validation layers requested but not available\n");
        arrFree(&layers);
        arrFree(&instance_extensions);
        return NULL;
    }

    VkApplicationInfo app_info;
    memset(&app_info, 0, sizeof(app_info));
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = NULL;
    app_info.pApplicationName = "Rendergraph application";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "Rendergraph";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info;
    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.flags = 0;
    instance_info.pApplicationInfo = &app_info;

    instance_info.enabledLayerCount = layers.len;
    instance_info.ppEnabledLayerNames = layers.ptr;

    switch (device->info.window_system)
    {
        case RG_WINDOW_SYSTEM_NONE: break;

        case RG_WINDOW_SYSTEM_WIN32:
        {
            arrPush(&instance_extensions, "VK_KHR_surface");
            arrPush(&instance_extensions, "VK_KHR_win32_surface");
            break;
        }
        case RG_WINDOW_SYSTEM_X11:
        {
            arrPush(&instance_extensions, "VK_KHR_surface");
            arrPush(&instance_extensions, "VK_KHR_xlib_surface");
            break;
        }
        case RG_WINDOW_SYSTEM_WAYLAND:
        {
            arrPush(&instance_extensions, "VK_KHR_surface");
            arrPush(&instance_extensions, "VK_KHR_wayland_surface");
            break;
        }
    }

    instance_info.enabledExtensionCount = instance_extensions.len;
    instance_info.ppEnabledExtensionNames = (const char *const *)instance_extensions.ptr;

    VK_CHECK(vkCreateInstance(&instance_info, NULL, &device->instance));

    arrFree(&layers);
    arrFree(&instance_extensions);

    volkLoadInstance(device->instance);

    if (device->info.enable_validation)
    {
        VkDebugUtilsMessengerCreateInfoEXT debug_create_info;
        memset(&debug_create_info, 0, sizeof(debug_create_info));
        debug_create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debug_create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debug_create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debug_create_info.pfnUserCallback = &debug_message_callback;

        VK_CHECK(vkCreateDebugUtilsMessengerEXT(
            device->instance, &debug_create_info, NULL, &device->debug_callback));
    }

    uint32_t num_physical_devices = 0;
    vkEnumeratePhysicalDevices(device->instance, &num_physical_devices, NULL);
    VkPhysicalDevice *physical_devices =
        (VkPhysicalDevice *)malloc(sizeof(VkPhysicalDevice) * num_physical_devices);
    vkEnumeratePhysicalDevices(device->instance, &num_physical_devices, physical_devices);

    if (num_physical_devices == 0)
    {
        fprintf(stderr, "No physical devices found\n");
        exit(1);
    }

    device->physical_device = physical_devices[0];

    free(physical_devices);

    vkGetPhysicalDeviceProperties(
        device->physical_device, &device->physical_device_properties);
    vkGetPhysicalDeviceFeatures(
        device->physical_device, &device->physical_device_features);

    vkGetPhysicalDeviceQueueFamilyProperties(
        device->physical_device, &device->num_queue_family_properties, NULL);
    device->queue_family_properties = (VkQueueFamilyProperties *)malloc(
        sizeof(VkQueueFamilyProperties) * device->num_queue_family_properties);
    vkGetPhysicalDeviceQueueFamilyProperties(
        device->physical_device,
        &device->num_queue_family_properties,
        device->queue_family_properties);

    fprintf(
        stderr,
        "Using physical device: %s\n",
        device->physical_device_properties.deviceName);
    VkPhysicalDeviceFeatures enabled_features;
    memset(&enabled_features, 0, sizeof(enabled_features));

    if (device->physical_device_features.samplerAnisotropy)
    {
        enabled_features.samplerAnisotropy = VK_TRUE;
    }

    if (device->physical_device_features.fillModeNonSolid)
    {
        enabled_features.fillModeNonSolid = VK_TRUE;
    }

    VkQueueFlags requested_queue_types = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;

    VkDeviceQueueCreateInfo queue_create_infos[2];
    uint32_t num_queue_create_infos = 0;

    // Get queue family indices for the requested queue family types
    // Note that the indices may overlap depending on the implementation

    const float default_queue_priority = 0.0f;

    // Graphics queue
    if (requested_queue_types & VK_QUEUE_GRAPHICS_BIT)
    {
        device->queue_family_indices.graphics =
            get_queue_family_index(device, VK_QUEUE_GRAPHICS_BIT);

        VkDeviceQueueCreateInfo queue_info;
        memset(&queue_info, 0, sizeof(queue_info));
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = device->queue_family_indices.graphics;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &default_queue_priority;

        queue_create_infos[num_queue_create_infos++] = queue_info;
    }
    else
    {
        device->queue_family_indices.graphics = 0;
    }

    // Dedicated compute queue
    if (requested_queue_types & VK_QUEUE_COMPUTE_BIT)
    {
        device->queue_family_indices.compute =
            get_queue_family_index(device, VK_QUEUE_COMPUTE_BIT);
        if (device->queue_family_indices.compute != device->queue_family_indices.graphics)
        {
            // If compute family index differs,
            // we need an additional queue create info for the compute queue
            VkDeviceQueueCreateInfo queue_info;
            memset(&queue_info, 0, sizeof(queue_info));
            queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queue_info.queueFamilyIndex = device->queue_family_indices.compute;
            queue_info.queueCount = 1;
            queue_info.pQueuePriorities = &default_queue_priority;

            queue_create_infos[num_queue_create_infos++] = queue_info;
        }
    }
    else
    {
        // Else we use the same queue
        device->queue_family_indices.compute = device->queue_family_indices.graphics;
    }

    // Create the logical device representation
    VkDeviceCreateInfo device_create_info;
    memset(&device_create_info, 0, sizeof(device_create_info));
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.queueCreateInfoCount = num_queue_create_infos;
    device_create_info.pQueueCreateInfos = queue_create_infos;
    device_create_info.pEnabledFeatures = &enabled_features;

    // Get available device extensions
    uint32_t num_available_device_extensions = 0;
    vkEnumerateDeviceExtensionProperties(
        device->physical_device, NULL, &num_available_device_extensions, NULL);

    VkExtensionProperties *available_device_extensions = (VkExtensionProperties *)malloc(
            sizeof(VkExtensionProperties) * num_available_device_extensions);
    vkEnumerateDeviceExtensionProperties(
        device->physical_device,
        NULL,
        &num_available_device_extensions,
        available_device_extensions);

    ARRAY_OF(const char*) device_extensions;
    memset(&device_extensions, 0, sizeof(device_extensions));

    if (device->info.window_system != RG_WINDOW_SYSTEM_NONE)
    {
        arrPush(&device_extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }

    for (uint32_t i = 0; i < device_extensions.len; ++i)
    {
        bool found = false;
        for (uint32_t j = 0; j < num_available_device_extensions; ++j)
        {
            if (strcmp(
                    available_device_extensions[j].extensionName,
                    device_extensions.ptr[i]) == 0)
            {
                found = true;
            }
        }

        if (!found)
        {
            arrFree(&device_extensions);
            free(available_device_extensions);

            fprintf(
                stderr,
                "required device extension not found: %s\n",
                device_extensions.ptr[i]);
            return NULL;
        }
    }

    device_create_info.enabledExtensionCount = device_extensions.len;
    device_create_info.ppEnabledExtensionNames =
        (const char *const *)device_extensions.ptr;

    VK_CHECK(vkCreateDevice(
        device->physical_device, &device_create_info, NULL, &device->device));

    arrFree(&device_extensions);
    free(available_device_extensions);

    // Fetch device memory properties
    vkGetPhysicalDeviceMemoryProperties(
        device->physical_device,
        &device->physical_device_memory_properties);

    vkGetDeviceQueue(
        device->device,
        device->queue_family_indices.graphics,
        0,
        &device->graphics_queue);

    // Initialize allocator
    device->allocator = rgAllocatorCreate(device);

    return device;
}

void rgDeviceDestroy(RgDevice *device)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));

    rgAllocatorDestroy(device->allocator);

    vkDestroyDevice(device->device, NULL);

    if (device->info.enable_validation)
    {
        vkDestroyDebugUtilsMessengerEXT(device->instance, device->debug_callback, NULL);
    }

    vkDestroyInstance(device->instance, NULL);

    free(device->queue_family_properties);

    free(device);
}

void rgDeviceWaitIdle(RgDevice* device)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));
}

RgFormat rgDeviceGetSupportedDepthFormat(RgDevice* device, RgFormat wanted_format)
{
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(
        device->physical_device,
        format_to_vk(wanted_format),
        &formatProperties);

    if (formatProperties.optimalTilingFeatures &
        (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
         VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
    {
        return wanted_format;
    }

    VkFormat depth_formats[5] = {
        VK_FORMAT_D32_SFLOAT_S8_UINT,
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D16_UNORM_S8_UINT,
        VK_FORMAT_D16_UNORM,
    };

    for (uint32_t i = 0; i < RG_LENGTH(depth_formats); ++i)
    {
        VkFormat format = depth_formats[i];

        vkGetPhysicalDeviceFormatProperties(
            device->physical_device,
            format,
            &formatProperties);

        if (formatProperties.optimalTilingFeatures &
            (VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
             VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
        {
            switch (format)
            {
            case VK_FORMAT_D32_SFLOAT_S8_UINT: return RG_FORMAT_D32_SFLOAT_S8_UINT;
            case VK_FORMAT_D32_SFLOAT: return RG_FORMAT_D32_SFLOAT;
            case VK_FORMAT_D24_UNORM_S8_UINT: return RG_FORMAT_D24_UNORM_S8_UINT;
            case VK_FORMAT_D16_UNORM_S8_UINT: return RG_FORMAT_D16_UNORM_S8_UINT;
            case VK_FORMAT_D16_UNORM: return RG_FORMAT_D16_UNORM;
            default: assert(0); break;
            }
        }
    }
    
    return RG_FORMAT_UNDEFINED;
}

void rgObjectSetName(RgDevice *device, RgObjectType type, void *object, const char* name)
{
    if (device->info.enable_validation)
    {
        VkDebugUtilsObjectNameInfoEXT info;
        memset(&info, 0, sizeof(info));
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
        info.pObjectName = name;

        switch (type)
        {
        case RG_OBJECT_TYPE_IMAGE: {
            info.objectType = VK_OBJECT_TYPE_IMAGE;
            info.objectHandle = (uint64_t)(((RgImage*)object)->image);
            break;
        }

        case RG_OBJECT_TYPE_BUFFER: {
            info.objectType = VK_OBJECT_TYPE_BUFFER;
            info.objectHandle = (uint64_t)(((RgBuffer*)object)->buffer);
            break;
        }

        case RG_OBJECT_TYPE_UNKNOWN: assert(0); break;
        }

        VK_CHECK(vkSetDebugUtilsObjectNameEXT(device->device, &info));
    }
}

RgCmdPool *rgCmdPoolCreate(RgDevice* device)
{
    RgCmdPool *cmd_pool = (RgCmdPool *)malloc(sizeof(*cmd_pool));
    memset(cmd_pool, 0, sizeof(*cmd_pool));

    VkCommandPoolCreateInfo cmd_pool_info;
    memset(&cmd_pool_info, 0, sizeof(cmd_pool_info));
    cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cmd_pool_info.queueFamilyIndex = device->queue_family_indices.graphics;
    cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(
        device->device, &cmd_pool_info, NULL, &cmd_pool->command_pool));

    return cmd_pool;
}

void rgCmdPoolDestroy(RgDevice* device, RgCmdPool *cmd_pool)
{
    vkDestroyCommandPool(device->device, cmd_pool->command_pool, NULL);
    free(cmd_pool);
}
// }}}

// Swapchain setup {{{
static void
rgSwapchainInit(
    RgDevice *device,
    RgSwapchain *swapchain,
    RgPlatformWindowInfo *window,
    RgFormat preferred_format)
{
    memset(swapchain, 0, sizeof(*swapchain));

    swapchain->window = *window;
    swapchain->device = device;
    swapchain->preferred_format = preferred_format;
    assert(swapchain->preferred_format != RG_FORMAT_UNDEFINED);

    switch (device->info.window_system)
    {
        case RG_WINDOW_SYSTEM_NONE: assert(0); break;

        case RG_WINDOW_SYSTEM_WIN32:
        {
#if defined(_WIN32)
            VkWin32SurfaceCreateInfoKHR surface_ci;
            memset(&surface_ci, 0, sizeof(surface_ci));
            surface_ci.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
            surface_ci.hinstance = GetModuleHandle(NULL);
            surface_ci.hwnd = (HWND)window->win32.window;

            VK_CHECK(vkCreateWin32SurfaceKHR(
                        device->instance,
                        &surface_ci,
                        NULL,
                        &swapchain->surface));
#else
            assert(0);
#endif
            break;
        }
        case RG_WINDOW_SYSTEM_X11:
        {
#if defined(__linux__)
            VkXlibSurfaceCreateInfoKHR surface_ci;
            memset(&surface_ci, 0, sizeof(surface_ci));
            surface_ci.sType = VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR;
            surface_ci.dpy = (Display *)window->x11.display;
            surface_ci.window = (Window)window->x11.window;

            VK_CHECK(vkCreateXlibSurfaceKHR(
                         device->instance,
                         &surface_ci,
                         NULL,
                         &swapchain->surface));
#else
            assert(0);
#endif
            break;
        }
        case RG_WINDOW_SYSTEM_WAYLAND:
        {
#if defined(__linux__)
            VkWaylandSurfaceCreateInfoKHR surface_ci;
            memset(&surface_ci, 0, sizeof(surface_ci));
            surface_ci.sType = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
            surface_ci.display = (struct wl_display*)window->wl.display;
            surface_ci.surface = (struct wl_surface*)window->wl.window;

            VK_CHECK(vkCreateWaylandSurfaceKHR(
                         device->instance,
                         &surface_ci,
                         NULL,
                         &swapchain->surface));
#else
            assert(0);
#endif
            break;
        }
    }

    swapchain->present_family_index = UINT32_MAX;

    uint32_t num_queue_families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        device->physical_device, &num_queue_families, NULL);
    VkQueueFamilyProperties *queue_families = (VkQueueFamilyProperties *)malloc(
        sizeof(VkQueueFamilyProperties) * num_queue_families);
    vkGetPhysicalDeviceQueueFamilyProperties(
        device->physical_device, &num_queue_families, queue_families);

    for (uint32_t i = 0; i < num_queue_families; ++i)
    {
        VkQueueFamilyProperties *queue_family = &queue_families[i];

        VkBool32 supported;
        vkGetPhysicalDeviceSurfaceSupportKHR(
            device->physical_device, i, swapchain->surface, &supported);

        if (queue_family->queueCount > 0 && supported)
        {
            swapchain->present_family_index = i;
            break;
        }
    }

    free(queue_families);

    if (swapchain->present_family_index == UINT32_MAX)
    {
        fprintf(stderr, "Could not obtain a present queue family.\n");
        exit(1);
    }

    // Get present queue
    vkGetDeviceQueue(
        device->device, swapchain->present_family_index, 0, &swapchain->present_queue);
}

static void rgSwapchainDestroy(RgDevice *device, RgSwapchain *swapchain)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));

    for (uint32_t i = 0; i < swapchain->num_images; ++i)
    {
        vkDestroyImageView(device->device, swapchain->image_views[i], NULL);
        swapchain->image_views[i] = VK_NULL_HANDLE;
    }

    vkDestroySwapchainKHR(device->device, swapchain->swapchain, NULL);
    vkDestroySurfaceKHR(device->instance, swapchain->surface, NULL);

    free(swapchain->images);
    free(swapchain->image_views);
}

static void rgSwapchainResize(RgSwapchain *swapchain, uint32_t width, uint32_t height)
{
    VK_CHECK(vkDeviceWaitIdle(swapchain->device->device));

    VkSwapchainKHR old_swapchain = swapchain->swapchain;
    uint32_t old_num_images = swapchain->num_images;
    VkImageView *old_image_views = swapchain->image_views;

    swapchain->swapchain = VK_NULL_HANDLE;
    swapchain->num_images = 0;
    swapchain->image_views = NULL;

    // Get format
    VkSurfaceFormatKHR surface_format;
    memset(&surface_format, 0, sizeof(surface_format));
    {
        uint32_t num_formats = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            swapchain->device->physical_device, swapchain->surface, &num_formats, NULL);
        VkSurfaceFormatKHR *formats =
            (VkSurfaceFormatKHR *)malloc(sizeof(VkSurfaceFormatKHR) * num_formats);
        vkGetPhysicalDeviceSurfaceFormatsKHR(
            swapchain->device->physical_device,
            swapchain->surface,
            &num_formats,
            formats);

        if (num_formats == 0)
        {
            fprintf(stderr, "Physical device does not support swapchain creation\n");
            exit(1);
        }

        if (num_formats == 1 && formats[0].format == VK_FORMAT_UNDEFINED)
        {
            surface_format.format = VK_FORMAT_B8G8R8A8_UNORM;
            surface_format.colorSpace = formats[0].colorSpace;
        }

        for (uint32_t i = 0; i < num_formats; ++i)
        {
            VkSurfaceFormatKHR *format = &formats[i];
            if (format->format == format_to_vk(swapchain->preferred_format))
            {
                surface_format = *format;
                break;
            }
        }

        free(formats);
    }

    swapchain->image_format = surface_format.format;

    // Get present mode
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    {
        uint32_t num_present_modes = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            swapchain->device->physical_device,
            swapchain->surface,
            &num_present_modes,
            NULL);
        VkPresentModeKHR *present_modes =
            (VkPresentModeKHR *)malloc(sizeof(VkPresentModeKHR) * num_present_modes);
        vkGetPhysicalDeviceSurfacePresentModesKHR(
            swapchain->device->physical_device,
            swapchain->surface,
            &num_present_modes,
            present_modes);

        if (num_present_modes == 0)
        {
            fprintf(stderr, "Physical device does not support swapchain creation\n");
            exit(1);
        }

        for (uint32_t i = 0; i < num_present_modes; ++i)
        {
            if (present_modes[i] == VK_PRESENT_MODE_FIFO_KHR)
            {
                present_mode = present_modes[i];
                break;
            }

            if (present_modes[i] == VK_PRESENT_MODE_FIFO_RELAXED_KHR)
            {
                present_mode = present_modes[i];
                break;
            }

            if (present_modes[i] == VK_PRESENT_MODE_MAILBOX_KHR)
            {
                present_mode = present_modes[i];
                break;
            }

            if (present_modes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
            {
                present_mode = present_modes[i];
                break;
            }
        }

        free(present_modes);
    }

    // Get capabilities
    VkSurfaceCapabilitiesKHR surface_capabilities = {0};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        swapchain->device->physical_device, swapchain->surface, &surface_capabilities);

    if (surface_capabilities.currentExtent.width == (uint32_t)-1)
    {
        // If the surface size is undefined, the size is set to
        // the size of the images requested.
        swapchain->extent.width = RG_CLAMP(
            width,
            surface_capabilities.minImageExtent.width,
            surface_capabilities.maxImageExtent.width);
        swapchain->extent.height = RG_CLAMP(
            height,
            surface_capabilities.minImageExtent.height,
            surface_capabilities.maxImageExtent.height);
    }
    else
    {
        // If the surface size is defined, the swap chain size must match
        swapchain->extent = surface_capabilities.currentExtent;
    }

    VkImageUsageFlags image_usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!(surface_capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
    {
        fprintf(
            stderr,
            "Physical device does not support "
            "VK_IMAGE_USAGE_TRANSFER_DST_BIT in swapchains\n");
        exit(1);
    }

    VkSwapchainCreateInfoKHR create_info;
    memset(&create_info, 0, sizeof(create_info));
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = swapchain->surface;
    create_info.minImageCount = RG_CLAMP(2, surface_capabilities.minImageCount, surface_capabilities.maxImageCount);
    create_info.imageFormat = surface_format.format;
    create_info.imageColorSpace = surface_format.colorSpace;
    create_info.imageExtent = swapchain->extent;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = image_usage;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = 0;
    create_info.pQueueFamilyIndices = NULL;
    create_info.preTransform = surface_capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = old_swapchain;

    VK_CHECK(vkCreateSwapchainKHR(
        swapchain->device->device, &create_info, NULL, &swapchain->swapchain));

    vkGetSwapchainImagesKHR(
        swapchain->device->device, swapchain->swapchain, &swapchain->num_images, NULL);
    swapchain->images =
        (VkImage *)realloc(swapchain->images, sizeof(VkImage) * swapchain->num_images);
    vkGetSwapchainImagesKHR(
        swapchain->device->device,
        swapchain->swapchain,
        &swapchain->num_images,
        swapchain->images);

    swapchain->image_views = (VkImageView *)realloc(
        swapchain->image_views, sizeof(VkImageView) * swapchain->num_images);
    for (size_t i = 0; i < swapchain->num_images; i++)
    {
        VkImageViewCreateInfo view_create_info;
        memset(&view_create_info, 0, sizeof(view_create_info));
        view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_create_info.image = swapchain->images[i];
        view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_create_info.format = swapchain->image_format;
        view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_create_info.subresourceRange.baseMipLevel = 0;
        view_create_info.subresourceRange.levelCount = 1;
        view_create_info.subresourceRange.baseArrayLayer = 0;
        view_create_info.subresourceRange.layerCount = 1;

        VK_CHECK(vkCreateImageView(
            swapchain->device->device,
            &view_create_info,
            NULL,
            &swapchain->image_views[i]));
    }

    // Destroy old stuff
    if (old_image_views)
    {
        for (uint32_t i = 0; i < old_num_images; ++i)
        {
            vkDestroyImageView(
                swapchain->device->device, old_image_views[i], NULL);
        }
        free(old_image_views);
    }

    if (old_swapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(swapchain->device->device, old_swapchain, NULL);
    }
}
// }}}

// Buffer {{{
RgBuffer *rgBufferCreate(RgDevice *device, RgBufferInfo *info)
{
    RgBuffer *buffer = (RgBuffer *)malloc(sizeof(RgBuffer));
    memset(buffer, 0, sizeof(*buffer));

    buffer->info = *info;

    assert(buffer->info.size > 0);
    assert(buffer->info.memory > 0);
    assert(buffer->info.usage > 0);

    VkBufferCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = buffer->info.size;

    if (buffer->info.usage & RG_BUFFER_USAGE_VERTEX)
        ci.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    if (buffer->info.usage & RG_BUFFER_USAGE_INDEX)
        ci.usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    if (buffer->info.usage & RG_BUFFER_USAGE_UNIFORM)
        ci.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (buffer->info.usage & RG_BUFFER_USAGE_TRANSFER_SRC)
        ci.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (buffer->info.usage & RG_BUFFER_USAGE_TRANSFER_DST)
        ci.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (buffer->info.usage & RG_BUFFER_USAGE_STORAGE)
        ci.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    VK_CHECK(vkCreateBuffer(
        device->device,
        &ci,
        NULL,
        &buffer->buffer));

    RgAllocationInfo alloc_info = {0};
    vkGetBufferMemoryRequirements(device->device, buffer->buffer, &alloc_info.requirements);

    switch (buffer->info.memory)
    {
    case RG_BUFFER_MEMORY_HOST: alloc_info.type = RG_ALLOCATION_TYPE_CPU_TO_GPU; break;
    case RG_BUFFER_MEMORY_DEVICE: alloc_info.type = RG_ALLOCATION_TYPE_GPU_ONLY; break;
    }

    alloc_info.dedicated = true;
    VK_CHECK(rgAllocatorAllocate(device->allocator, &alloc_info, &buffer->allocation));

    if (buffer->allocation.dedicated)
    {
        VK_CHECK(vkBindBufferMemory(
            device->device,
            buffer->buffer,
            buffer->allocation.dedicated_memory,
            buffer->allocation.offset));
    }
    else
    {
        VK_CHECK(vkBindBufferMemory(
            device->device,
            buffer->buffer,
            buffer->allocation.block->handle,
            buffer->allocation.offset));
    }

    return buffer;
}

void rgBufferDestroy(RgDevice *device, RgBuffer *buffer)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));
    if (buffer->buffer)
    {
        rgAllocatorFree(device->allocator, &buffer->allocation);
        vkDestroyBuffer(device->device, buffer->buffer, NULL);
    }

    free(buffer);
}

void *rgBufferMap(RgDevice *device, RgBuffer *buffer)
{
    void *ptr;
    VK_CHECK(rgMapAllocation(device->allocator, &buffer->allocation, &ptr));
    return ptr;
}

void rgBufferUnmap(RgDevice *device, RgBuffer *buffer)
{
    rgUnmapAllocation(device->allocator, &buffer->allocation);
}

void rgBufferUpload(
    RgDevice *device, RgCmdPool *cmd_pool, RgBuffer *buffer, size_t offset, size_t size, void *data)
{
    VkCommandBuffer cmd_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    RgBufferInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.size = size;
    buffer_info.usage = RG_BUFFER_USAGE_TRANSFER_SRC;
    buffer_info.memory = RG_BUFFER_MEMORY_HOST;

    RgBuffer *staging = rgBufferCreate(device, &buffer_info);

    void *staging_ptr = rgBufferMap(device, staging);
    memcpy(staging_ptr, data, size);
    rgBufferUnmap(device, staging);

    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(device->device, &fence_info, NULL, &fence));

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = cmd_pool->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VK_CHECK(vkAllocateCommandBuffers(device->device, &alloc_info, &cmd_buffer));

    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_buffer, &begin_info));

    VkBufferCopy region;
    memset(&region, 0, sizeof(region));
    region.srcOffset = 0;
    region.dstOffset = offset;
    region.size = size;
    vkCmdCopyBuffer(cmd_buffer, staging->buffer, buffer->buffer, 1, &region);

    VK_CHECK(vkEndCommandBuffer(cmd_buffer));

    VkSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd_buffer;

    VK_CHECK(vkQueueSubmit(device->graphics_queue, 1, &submit, fence));

    VK_CHECK(vkWaitForFences(device->device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(device->device, fence, NULL);

    vkFreeCommandBuffers(device->device, cmd_pool->command_pool, 1, &cmd_buffer);

    rgBufferDestroy(device, staging);
}
// }}}

// Image {{{
RgImage *rgImageCreate(RgDevice *device, RgImageInfo *info)
{
    RgImage *image = (RgImage *)malloc(sizeof(RgImage));
    memset(image, 0, sizeof(*image));

    image->info = *info;

    if (image->info.depth == 0) image->info.depth = 1;
    if (image->info.sample_count == 0) image->info.sample_count = 1;
    if (image->info.mip_count == 0) image->info.mip_count = 1;
    if (image->info.layer_count == 0) image->info.layer_count = 1;
    if (image->info.usage == 0)
        image->info.usage = RG_IMAGE_USAGE_SAMPLED | RG_IMAGE_USAGE_TRANSFER_DST;
    if (image->info.aspect == 0) image->info.aspect = RG_IMAGE_ASPECT_COLOR;

    assert(image->info.width > 0);
    assert(image->info.height > 0);
    assert(image->info.format != RG_FORMAT_UNDEFINED);

    {
        VkImageCreateInfo ci;
        memset(&ci, 0, sizeof(ci));
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = format_to_vk(image->info.format);
        ci.extent.width = image->info.width;
        ci.extent.height = image->info.height;
        ci.extent.depth = image->info.depth;
        ci.mipLevels = image->info.mip_count;
        ci.arrayLayers = image->info.layer_count;
        ci.samples = (VkSampleCountFlagBits)image->info.sample_count;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;

        if (image->info.layer_count == 6)
        {
            ci.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        }

        if (image->info.usage & RG_IMAGE_USAGE_SAMPLED)
            ci.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        if (image->info.usage & RG_IMAGE_USAGE_TRANSFER_DST)
            ci.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (image->info.usage & RG_IMAGE_USAGE_TRANSFER_SRC)
            ci.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (image->info.usage & RG_IMAGE_USAGE_STORAGE)
            ci.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        if (image->info.usage & RG_IMAGE_USAGE_COLOR_ATTACHMENT)
            ci.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (image->info.usage & RG_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT)
            ci.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

        VK_CHECK(vkCreateImage(device->device, &ci, NULL, &image->image));

        VkMemoryDedicatedRequirements dedicated_requirements = {0};
        dedicated_requirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;

        VkMemoryRequirements2 memory_requirements = {0};
        memory_requirements.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
        memory_requirements.pNext = &dedicated_requirements;

        VkImageMemoryRequirementsInfo2 image_requirements_info = {0};
        image_requirements_info.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2;
        image_requirements_info.image = image->image;

        vkGetImageMemoryRequirements2(device->device, &image_requirements_info, &memory_requirements);

        RgAllocationInfo alloc_info = {0};
        alloc_info.type = RG_ALLOCATION_TYPE_GPU_ONLY;
        alloc_info.requirements = memory_requirements.memoryRequirements;
        alloc_info.dedicated = dedicated_requirements.prefersDedicatedAllocation ||
            dedicated_requirements.requiresDedicatedAllocation;
        alloc_info.dedicated = true;

        VK_CHECK(rgAllocatorAllocate(device->allocator, &alloc_info, &image->allocation));

        if (image->allocation.dedicated)
        {
            VK_CHECK(vkBindImageMemory(
                        device->device,
                        image->image,
                        image->allocation.dedicated_memory,
                        image->allocation.offset));
        }
        else
        {
            VK_CHECK(vkBindImageMemory(
                        device->device,
                        image->image,
                        image->allocation.block->handle,
                        image->allocation.offset));
        }
    }

    {
        VkImageViewCreateInfo ci;
        memset(&ci, 0, sizeof(ci));
        ci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        ci.image = image->image;
        ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        ci.format = format_to_vk(image->info.format);
        ci.subresourceRange.baseMipLevel = 0;
        ci.subresourceRange.levelCount = image->info.mip_count;
        ci.subresourceRange.baseArrayLayer = 0;
        ci.subresourceRange.layerCount = image->info.layer_count;

        if (image->info.layer_count == 6)
        {
            ci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
        }

        if (image->info.aspect & RG_IMAGE_ASPECT_COLOR)
            image->aspect |= VK_IMAGE_ASPECT_COLOR_BIT;
        if (image->info.aspect & RG_IMAGE_ASPECT_DEPTH)
            image->aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
        if (image->info.aspect & RG_IMAGE_ASPECT_STENCIL)
        {
            switch (image->info.format)
            {
            case RG_FORMAT_D16_UNORM_S8_UINT:
            case RG_FORMAT_D24_UNORM_S8_UINT:
            case RG_FORMAT_D32_SFLOAT_S8_UINT:
                image->aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
                break;
            default: break;
            }
        }

        ci.subresourceRange.aspectMask = image->aspect;

        VK_CHECK(vkCreateImageView(device->device, &ci, NULL, &image->view));
    }

    return image;
}

void rgImageDestroy(RgDevice *device, RgImage *image)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));
    if (image->view)
    {
        vkDestroyImageView(device->device, image->view, NULL);
    }
    if (image->image)
    {
        vkDestroyImage(device->device, image->image, NULL);
        rgAllocatorFree(device->allocator, &image->allocation);
    }

    free(image);
}

void rgImageUpload(
    RgDevice *device, RgCmdPool *cmd_pool, RgImageCopy *dst, RgExtent3D *extent, size_t size, void *data)
{
    VkCommandBuffer cmd_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    RgBufferInfo buffer_info;
    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.size = size;
    buffer_info.usage = RG_BUFFER_USAGE_TRANSFER_SRC;
    buffer_info.memory = RG_BUFFER_MEMORY_HOST;

    RgBuffer *staging = rgBufferCreate(device, &buffer_info);

    void *staging_ptr = rgBufferMap(device, staging);
    memcpy(staging_ptr, data, size);
    rgBufferUnmap(device, staging);

    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(device->device, &fence_info, NULL, &fence));

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = cmd_pool->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VK_CHECK(vkAllocateCommandBuffers(device->device, &alloc_info, &cmd_buffer));

    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_buffer, &begin_info));

    VkImageSubresourceRange subresource_range;
    memset(&subresource_range, 0, sizeof(subresource_range));
    subresource_range.aspectMask = dst->image->aspect;
    subresource_range.baseMipLevel = dst->mip_level;
    subresource_range.levelCount = 1;
    subresource_range.baseArrayLayer = dst->array_layer;
    subresource_range.layerCount = 1;

    VkImageMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.image = dst->image->image;
    barrier.subresourceRange = subresource_range;

    vkCmdPipelineBarrier(
        cmd_buffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0,
        NULL,
        0,
        NULL,
        1,
        &barrier);

    VkBufferImageCopy region;
    memset(&region, 0, sizeof(region));
    region.imageSubresource.aspectMask = dst->image->aspect;
    region.imageSubresource.mipLevel = dst->mip_level;
    region.imageSubresource.baseArrayLayer = dst->array_layer;
    region.imageSubresource.layerCount = 1;
    region.imageOffset.x = dst->offset.x;
    region.imageOffset.y = dst->offset.y;
    region.imageOffset.z = dst->offset.z;
    region.imageExtent.width = extent->width;
    region.imageExtent.height = extent->height;
    region.imageExtent.depth = extent->depth;

    vkCmdCopyBufferToImage(
        cmd_buffer,
        staging->buffer,
        dst->image->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region);

    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    vkCmdPipelineBarrier(
        cmd_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0,
        0,
        NULL,
        0,
        NULL,
        1,
        &barrier);

    VK_CHECK(vkEndCommandBuffer(cmd_buffer));

    VkSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd_buffer;

    VK_CHECK(vkQueueSubmit(device->graphics_queue, 1, &submit, fence));

    VK_CHECK(vkWaitForFences(device->device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(device->device, fence, NULL);

    vkFreeCommandBuffers(device->device, cmd_pool->command_pool, 1, &cmd_buffer);

    rgBufferDestroy(device, staging);
}

void rgImageBarrier(
    RgDevice *device,
    RgCmdPool *cmd_pool, 
    RgImage *image,
    const RgImageRegion *region,
    RgResourceUsage from,
    RgResourceUsage to)
{
    VkCommandBuffer cmd_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(device->device, &fence_info, NULL, &fence));

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = cmd_pool->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VK_CHECK(vkAllocateCommandBuffers(device->device, &alloc_info, &cmd_buffer));

    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_buffer, &begin_info));

    VkImageSubresourceRange subresource_range;
    memset(&subresource_range, 0, sizeof(subresource_range));
    subresource_range.aspectMask = image->aspect;
    subresource_range.baseMipLevel = region->base_mip_level;
    subresource_range.levelCount = region->mip_count;
    subresource_range.baseArrayLayer = region->base_array_layer;
    subresource_range.layerCount = region->layer_count;

    VkImageMemoryBarrier barrier;
    memset(&barrier, 0, sizeof(barrier));
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image->image;
    barrier.subresourceRange = subresource_range;

    rgResourceUsageToVk(from, &barrier.srcAccessMask, &barrier.oldLayout);
    rgResourceUsageToVk(to, &barrier.dstAccessMask, &barrier.newLayout);

    vkCmdPipelineBarrier(
        cmd_buffer,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0,
        0, NULL,
        0, NULL,
        1, &barrier);

    VK_CHECK(vkEndCommandBuffer(cmd_buffer));

    VkSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd_buffer;

    VK_CHECK(vkQueueSubmit(device->graphics_queue, 1, &submit, fence));

    VK_CHECK(vkWaitForFences(device->device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(device->device, fence, NULL);

    vkFreeCommandBuffers(device->device, cmd_pool->command_pool, 1, &cmd_buffer);
}

void rgImageGenerateMipMaps(RgDevice *device, RgCmdPool *cmd_pool, RgImage *image)
{
    VkCommandBuffer cmd_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    VkFenceCreateInfo fence_info;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VK_CHECK(vkCreateFence(device->device, &fence_info, NULL, &fence));

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = cmd_pool->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VK_CHECK(vkAllocateCommandBuffers(device->device, &alloc_info, &cmd_buffer));

    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_buffer, &begin_info));

    VkImageSubresourceRange subresource_range;
    memset(&subresource_range, 0, sizeof(subresource_range));
    subresource_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresource_range.levelCount = 1;
    subresource_range.layerCount = 1;

    for (uint32_t i = 1; i < image->info.mip_count; i++) {
        VkImageBlit image_blit;
        memset(&image_blit, 0, sizeof(image_blit));

        image_blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_blit.srcSubresource.layerCount = 1;
        image_blit.srcSubresource.mipLevel = i - 1;
        image_blit.srcOffsets[1].x = (int32_t)(image->info.width >> (i - 1));
        image_blit.srcOffsets[1].y = (int32_t)(image->info.height >> (i - 1));
        image_blit.srcOffsets[1].z = 1;

        image_blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        image_blit.dstSubresource.layerCount = 1;
        image_blit.dstSubresource.mipLevel = i;
        image_blit.dstOffsets[1].x = (int32_t)(image->info.width >> i);
        image_blit.dstOffsets[1].y = (int32_t)(image->info.height >> i);
        image_blit.dstOffsets[1].z = 1;

        VkImageSubresourceRange mip_sub_range;
        memset(&mip_sub_range, 0, sizeof(mip_sub_range));
        mip_sub_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        mip_sub_range.baseMipLevel = i;
        mip_sub_range.levelCount = 1;
        mip_sub_range.layerCount = 1;

        {
            VkImageMemoryBarrier image_memory_barrier;
            memset(&image_memory_barrier, 0, sizeof(image_memory_barrier));
            image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            image_memory_barrier.srcAccessMask = 0;
            image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            image_memory_barrier.image = image->image;
            image_memory_barrier.subresourceRange = mip_sub_range;
            vkCmdPipelineBarrier(
                cmd_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, NULL,
                0, NULL,
                1, &image_memory_barrier);
        }

        vkCmdBlitImage(
            cmd_buffer,
            image->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &image_blit,
            VK_FILTER_LINEAR);

        {
            VkImageMemoryBarrier image_memory_barrier;
            memset(&image_memory_barrier, 0, sizeof(image_memory_barrier));
            image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            image_memory_barrier.image = image->image;
            image_memory_barrier.subresourceRange = mip_sub_range;
            vkCmdPipelineBarrier(
                cmd_buffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0, NULL,
                0, NULL,
                1, &image_memory_barrier);
        }
    }

    subresource_range.levelCount = image->info.mip_count;

    {
        VkImageMemoryBarrier image_memory_barrier;
        memset(&image_memory_barrier, 0, sizeof(image_memory_barrier));
        image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        image_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        image_memory_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_memory_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        image_memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        image_memory_barrier.image = image->image;
        image_memory_barrier.subresourceRange = subresource_range;
        vkCmdPipelineBarrier(
            cmd_buffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0,
            0, NULL,
            0, NULL,
            1, &image_memory_barrier);
    }

    VK_CHECK(vkEndCommandBuffer(cmd_buffer));

    VkSubmitInfo submit;
    memset(&submit, 0, sizeof(submit));
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd_buffer;

    VK_CHECK(vkQueueSubmit(device->graphics_queue, 1, &submit, fence));

    VK_CHECK(vkWaitForFences(device->device, 1, &fence, VK_TRUE, UINT64_MAX));
    vkDestroyFence(device->device, fence, NULL);

    vkFreeCommandBuffers(device->device, cmd_pool->command_pool, 1, &cmd_buffer);
}
// }}}

// Sampler {{{
struct RgSampler
{
    RgSamplerInfo info;
    VkSampler sampler;
};

RgSampler *rgSamplerCreate(RgDevice *device, RgSamplerInfo *info)
{
    RgSampler *sampler = (RgSampler *)malloc(sizeof(RgSampler));
    memset(sampler, 0, sizeof(*sampler));

    sampler->info = *info;

    if (sampler->info.min_lod == 0.0f && sampler->info.max_lod == 0.0f)
    {
        sampler->info.max_lod = 1.0f;
    }

    if (sampler->info.max_anisotropy == 0.0f)
    {
        sampler->info.max_anisotropy = 1.0f;
    }

    assert(sampler->info.max_lod >= sampler->info.min_lod);

    VkSamplerCreateInfo ci;
    memset(&ci, 0, sizeof(ci));
    ci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    ci.magFilter = filter_to_vk(sampler->info.mag_filter);
    ci.minFilter = filter_to_vk(sampler->info.min_filter);
    ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    ci.addressModeU = address_mode_to_vk(sampler->info.address_mode);
    ci.addressModeV = address_mode_to_vk(sampler->info.address_mode);
    ci.addressModeW = address_mode_to_vk(sampler->info.address_mode);
    ci.minLod = sampler->info.min_lod;
    ci.maxLod = sampler->info.max_lod;
    ci.maxAnisotropy = sampler->info.max_anisotropy;
    ci.anisotropyEnable = (VkBool32)sampler->info.anisotropy;
    ci.borderColor = border_color_to_vk(sampler->info.border_color);
    VK_CHECK(vkCreateSampler(device->device, &ci, NULL, &sampler->sampler));

    return sampler;
}

void rgSamplerDestroy(RgDevice *device, RgSampler *sampler)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));
    if (sampler->sampler)
    {
        vkDestroySampler(device->device, sampler->sampler, NULL);
    }
    sampler->sampler = VK_NULL_HANDLE;

    free(sampler);
}
// }}}

// Buffer allocator {{{
static RgBufferChunk *rgBufferChunkCreate(RgBufferPool *pool, size_t minimum_size)
{
    RgBufferChunk *chunk = (RgBufferChunk *)malloc(sizeof(*chunk));
    memset(chunk, 0, sizeof(*chunk));

    chunk->pool = pool;
    chunk->size = RG_MAX(minimum_size, pool->chunk_size);

    RgBufferInfo buffer_info = {0};
    buffer_info.size = chunk->size;
    buffer_info.usage = pool->usage;
    buffer_info.memory = RG_BUFFER_MEMORY_HOST;
    chunk->buffer = rgBufferCreate(chunk->pool->device, &buffer_info);

    chunk->mapping = (uint8_t *)rgBufferMap(pool->device, chunk->buffer);

    return chunk;
}

static void rgBufferChunkDestroy(RgBufferChunk *chunk)
{
    if (!chunk) return;
    rgBufferChunkDestroy(chunk->next);

    rgBufferUnmap(chunk->pool->device, chunk->buffer);
    rgBufferDestroy(chunk->pool->device, chunk->buffer);
    free(chunk);
}

static void rgBufferPoolInit(
    RgDevice *device,
    RgBufferPool *pool,
    size_t chunk_size,
    size_t alignment,
    RgBufferUsage usage)
{
    memset(pool, 0, sizeof(*pool));
    pool->device = device;
    pool->chunk_size = chunk_size;
    pool->alignment = alignment;
    pool->usage = usage;
}

static void rgBufferPoolReset(RgBufferPool *pool)
{
    RgBufferChunk *chunk = pool->base_chunk;
    while (chunk)
    {
        chunk->offset = 0;
        chunk = chunk->next;
    }
}

static RgBufferAllocation rgBufferPoolAllocate(RgBufferPool *pool, size_t allocate_size)
{
    RgBufferAllocation alloc = {0};

    RgBufferChunk *chunk = pool->base_chunk;
    RgBufferChunk *last_chunk = NULL;

buffer_pool_allocate_use_block:
    while (chunk)
    {
        size_t aligned_offset =
            (chunk->offset + pool->alignment - 1) & ~(pool->alignment - 1);
        if (chunk->mapping != NULL && chunk->size >= aligned_offset + allocate_size)
        {
            // Found chunk that fits the allocation
            assert(aligned_offset % pool->alignment == 0);
            chunk->offset = aligned_offset + allocate_size;

            alloc.buffer = chunk->buffer;
            alloc.mapping = chunk->mapping + aligned_offset;
            alloc.offset = aligned_offset;
            alloc.size = allocate_size;
            return alloc;
        }

        last_chunk = chunk;
        chunk = chunk->next;
    }

    // Did not find a chunk that fits the allocation

    RgBufferChunk *new_chunk = rgBufferChunkCreate(pool, allocate_size);
    if (last_chunk)
    {
        last_chunk->next = new_chunk;
    }
    else
    {
        pool->base_chunk = new_chunk;
    }

    chunk = new_chunk;
    goto buffer_pool_allocate_use_block;

    return alloc;
}

static void rgBufferPoolDestroy(RgBufferPool *pool)
{
    rgBufferChunkDestroy(pool->base_chunk);
}
// }}}

// Descriptor set allocator {{{
static void rgDescriptorPoolInit(
    RgDevice *device,
    RgDescriptorPool *pool,
    uint32_t num_bindings,
    VkDescriptorSetLayoutBinding *bindings)
{
    memset(pool, 0, sizeof(*pool));

    pool->device = device;
    pool->num_bindings = num_bindings;

    // Create set layout
    {
        VkDescriptorSetLayoutCreateInfo create_info;
        memset(&create_info, 0, sizeof(create_info));
        create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        create_info.bindingCount = num_bindings;
        create_info.pBindings = bindings;

        VK_CHECK(vkCreateDescriptorSetLayout(
            device->device, &create_info, NULL, &pool->set_layout));
    }

    // Create update template
    {
        VkDescriptorUpdateTemplateEntry entries[RG_MAX_DESCRIPTOR_BINDINGS];

        for (uint32_t b = 0; b < num_bindings; ++b)
        {
            VkDescriptorSetLayoutBinding *binding = &bindings[b];
            assert(b == binding->binding);

            VkDescriptorUpdateTemplateEntry entry;
            memset(&entry, 0, sizeof(entry));
            entry.dstBinding = binding->binding;
            entry.dstArrayElement = 0;
            entry.descriptorCount = binding->descriptorCount;
            entry.descriptorType = binding->descriptorType;
            entry.offset = binding->binding * sizeof(RgDescriptor);
            entry.stride = sizeof(RgDescriptor);

            entries[b] = entry;
        }

        VkDescriptorUpdateTemplateCreateInfo template_info;
        memset(&template_info, 0, sizeof(template_info));
        template_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO;
        template_info.descriptorUpdateEntryCount = num_bindings;
        template_info.pDescriptorUpdateEntries = entries;
        template_info.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET;
        template_info.descriptorSetLayout = pool->set_layout;

        VK_CHECK(vkCreateDescriptorUpdateTemplate(
            device->device, &template_info, NULL, &pool->update_template));
    }

    {
        for (uint32_t b = 0; b < num_bindings; ++b)
        {
            VkDescriptorSetLayoutBinding *binding = &bindings[b];

            VkDescriptorPoolSize *found_pool_size = NULL;

            for (uint32_t p = 0; p < pool->num_pool_sizes; ++p)
            {
                VkDescriptorPoolSize *pool_size = &pool->pool_sizes[p];
                if (pool_size->type == binding->descriptorType)
                {
                    found_pool_size = pool_size;
                    break;
                }
            }

            if (!found_pool_size)
            {
                VkDescriptorPoolSize new_pool_size = {binding->descriptorType, 0};
                pool->pool_sizes[pool->num_pool_sizes++] = new_pool_size;
                found_pool_size = &pool->pool_sizes[pool->num_pool_sizes - 1];
            }

            found_pool_size->descriptorCount += RG_SETS_PER_PAGE;
        }

        assert(pool->num_pool_sizes > 0);
    }
}

static void rgDescriptorPoolDestroy(RgDescriptorPool *pool)
{
    RgDescriptorPoolChunk *chunk = pool->base_chunk;
    while (chunk)
    {
        vkDestroyDescriptorPool(pool->device->device, chunk->pool, NULL);
        rgHashmapDestroy(&chunk->map);

        RgDescriptorPoolChunk *last_chunk = chunk;
        chunk = chunk->next;
        free(last_chunk);
    }

    vkDestroyDescriptorSetLayout(pool->device->device, pool->set_layout, NULL);
    vkDestroyDescriptorUpdateTemplate(pool->device->device, pool->update_template, NULL);
}

static void rgDescriptorPoolGrow(RgDescriptorPool *pool)
{
    RgDescriptorPoolChunk *chunk = (RgDescriptorPoolChunk *)malloc(sizeof(*chunk));
    memset(chunk, 0, sizeof(*chunk));

    chunk->next = pool->base_chunk;
    pool->base_chunk = chunk;

    rgHashmapInit(&chunk->map, RG_SETS_PER_PAGE);

    VkDescriptorPoolCreateInfo pool_create_info;
    memset(&pool_create_info, 0, sizeof(pool_create_info));
    pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_create_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_create_info.maxSets = RG_SETS_PER_PAGE;
    pool_create_info.poolSizeCount = pool->num_pool_sizes;
    pool_create_info.pPoolSizes = pool->pool_sizes;

    VK_CHECK(vkCreateDescriptorPool(
        pool->device->device, &pool_create_info, NULL, &chunk->pool));

    // Allocate descriptor sets
    VkDescriptorSetLayout set_layouts[RG_SETS_PER_PAGE];
    for (uint32_t i = 0; i < RG_SETS_PER_PAGE; i++)
    {
        set_layouts[i] = pool->set_layout;
    }

    VkDescriptorSetAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = chunk->pool;
    alloc_info.descriptorSetCount = RG_SETS_PER_PAGE;
    alloc_info.pSetLayouts = set_layouts;

    VK_CHECK(vkAllocateDescriptorSets(pool->device->device, &alloc_info, chunk->sets));
}

static VkDescriptorSet rgDescriptorPoolAllocate(
    RgDescriptorPool *pool, uint32_t num_descriptors, RgDescriptor *descriptors)
{
    assert(num_descriptors == pool->num_bindings);

    uint64_t descriptors_hash = 0;
    fnvHashReset(&descriptors_hash);
    fnvHashUpdate(
        &descriptors_hash,
        (uint8_t *)descriptors,
        sizeof(RgDescriptor) * num_descriptors);

    RgDescriptorPoolChunk *chunk = pool->base_chunk;
    while (chunk)
    {
        uint64_t *set_index_ptr = rgHashmapGet(&chunk->map, descriptors_hash);

        if (set_index_ptr != NULL)
        {
            // Set is available
            return chunk->sets[*set_index_ptr];
        }
        else
        {
            if (chunk->allocated_count >= RG_SETS_PER_PAGE)
            {
                // No sets available in this pool, so continue looking for another one
                continue;
            }

            // Update existing descriptor set, because we haven't found any
            // matching ones already

            uint64_t set_index = chunk->allocated_count;
            chunk->allocated_count++;

            VkDescriptorSet set = chunk->sets[set_index];
            rgHashmapSet(&chunk->map, descriptors_hash, set_index);

            vkUpdateDescriptorSetWithTemplate(
                pool->device->device, set, pool->update_template, descriptors);
            return set;
        }

        chunk = chunk->next;
    }

    rgDescriptorPoolGrow(pool);
    return rgDescriptorPoolAllocate(pool, num_descriptors, descriptors);
}
// }}}

// Pipeline {{{
RgPipeline *rgGraphicsPipelineCreate(RgDevice *device, RgGraphicsPipelineInfo *info)
{
    RgPipeline *pipeline = (RgPipeline *)malloc(sizeof(RgPipeline));
    memset(pipeline, 0, sizeof(*pipeline));

    pipeline->type = RG_PIPELINE_TYPE_GRAPHICS;

    pipeline->graphics.polygon_mode = info->polygon_mode;
    pipeline->graphics.cull_mode = info->cull_mode;
    pipeline->graphics.front_face = info->front_face;
    pipeline->graphics.topology = info->topology;
    pipeline->graphics.blend = info->blend;
    pipeline->graphics.depth_stencil = info->depth_stencil;

    pipeline->graphics.vertex_stride = info->vertex_stride;

    if (info->vertex_entry)
    {
        size_t str_size = strlen(info->vertex_entry) + 1;
        pipeline->graphics.vertex_entry = malloc(str_size);
        memcpy(pipeline->graphics.vertex_entry, info->vertex_entry, str_size);
    }
    if (info->fragment_entry)
    {
        size_t str_size = strlen(info->fragment_entry) + 1;
        pipeline->graphics.fragment_entry = malloc(str_size);
        memcpy(pipeline->graphics.fragment_entry, info->fragment_entry, str_size);
    }

    // Bindings
    pipeline->num_bindings = info->num_bindings;
    if (pipeline->num_bindings > 0)
    {
        pipeline->bindings =
            malloc(pipeline->num_bindings * sizeof(*pipeline->bindings));
        memcpy(
            pipeline->bindings,
            info->bindings,
            pipeline->num_bindings * sizeof(*pipeline->bindings));
    }

    // Vertex attributes
    pipeline->graphics.num_vertex_attributes = info->num_vertex_attributes;
    pipeline->graphics.vertex_attributes = malloc(
        pipeline->graphics.num_vertex_attributes *
        sizeof(*pipeline->graphics.vertex_attributes));
    memcpy(
        pipeline->graphics.vertex_attributes,
        info->vertex_attributes,
        pipeline->graphics.num_vertex_attributes *
        sizeof(*pipeline->graphics.vertex_attributes));

    rgHashmapInit(&pipeline->graphics.instances, 8);

    //
    // Create descriptor pools
    //

    VkDescriptorSetLayoutBinding bindings[RG_MAX_DESCRIPTOR_SETS]
                                         [RG_MAX_DESCRIPTOR_BINDINGS];
    uint32_t binding_counts[RG_MAX_DESCRIPTOR_SETS] = {0};
    uint32_t num_sets = 0;

    for (uint32_t i = 0; i < pipeline->num_bindings; ++i)
    {
        RgPipelineBinding *binding = &pipeline->bindings[i];
        VkDescriptorSetLayoutBinding *vk_binding =
            &bindings[binding->set][binding->binding];
        memset(vk_binding, 0, sizeof(*vk_binding));

        num_sets = RG_MAX(num_sets, binding->set + 1);
        binding_counts[binding->set] =
            RG_MAX(binding_counts[binding->set], binding->binding + 1);

        vk_binding->binding = binding->binding;
        vk_binding->descriptorType = pipeline_binding_type_to_vk(binding->type);
        vk_binding->descriptorCount = 1;
        vk_binding->stageFlags = VK_SHADER_STAGE_ALL; // TODO: this could be more specific
    }

    pipeline->num_sets = num_sets;

    VkDescriptorSetLayout set_layouts[RG_MAX_DESCRIPTOR_SETS];
    for (uint32_t i = 0; i < pipeline->num_sets; ++i)
    {
        rgDescriptorPoolInit(device, &pipeline->pools[i], binding_counts[i], bindings[i]);
        set_layouts[i] = pipeline->pools[i].set_layout;
    }

    //
    // Create pipeline layout
    //

    VkPipelineLayoutCreateInfo pipeline_layout_info;
    memset(&pipeline_layout_info, 0, sizeof(pipeline_layout_info));
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = pipeline->num_sets;
    pipeline_layout_info.pSetLayouts = set_layouts;
    pipeline_layout_info.pushConstantRangeCount = 0;
    pipeline_layout_info.pPushConstantRanges = NULL;

    VK_CHECK(vkCreatePipelineLayout(
        device->device, &pipeline_layout_info, NULL, &pipeline->pipeline_layout));

    if (info->vertex && info->vertex_size > 0)
    {
        VkShaderModuleCreateInfo module_create_info;
        memset(&module_create_info, 0, sizeof(module_create_info));
        module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_create_info.codeSize = info->vertex_size;
        module_create_info.pCode = (uint32_t *)info->vertex;

        VK_CHECK(vkCreateShaderModule(
            device->device, &module_create_info, NULL, &pipeline->graphics.vertex_shader));
    }

    if (info->fragment && info->fragment_size > 0)
    {
        VkShaderModuleCreateInfo module_create_info;
        memset(&module_create_info, 0, sizeof(module_create_info));
        module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        module_create_info.codeSize = info->fragment_size;
        module_create_info.pCode = (uint32_t *)info->fragment;

        VK_CHECK(vkCreateShaderModule(
            device->device, &module_create_info, NULL, &pipeline->graphics.fragment_shader));
    }

    return pipeline;
}

RgPipeline *rgComputePipelineCreate(RgDevice *device, RgComputePipelineInfo *info)
{
    RgPipeline *pipeline = (RgPipeline *)malloc(sizeof(RgPipeline));
    memset(pipeline, 0, sizeof(*pipeline));

    pipeline->type = RG_PIPELINE_TYPE_COMPUTE;
    pipeline->num_bindings = info->num_bindings;
    if (pipeline->num_bindings > 0)
    {
        pipeline->bindings =
            malloc(pipeline->num_bindings * sizeof(*pipeline->bindings));
        memcpy(
            pipeline->bindings,
            info->bindings,
            pipeline->num_bindings * sizeof(*pipeline->bindings));
    }

    assert(info->code && info->code_size > 0);

    //
    // Create descriptor pools
    //

    VkDescriptorSetLayoutBinding bindings[RG_MAX_DESCRIPTOR_SETS]
                                         [RG_MAX_DESCRIPTOR_BINDINGS];
    uint32_t binding_counts[RG_MAX_DESCRIPTOR_SETS] = {0};
    uint32_t num_sets = 0;

    for (uint32_t i = 0; i < pipeline->num_bindings; ++i)
    {
        RgPipelineBinding *binding = &pipeline->bindings[i];
        VkDescriptorSetLayoutBinding *vk_binding =
            &bindings[binding->set][binding->binding];
        memset(vk_binding, 0, sizeof(*vk_binding));

        num_sets = RG_MAX(num_sets, binding->set + 1);
        binding_counts[binding->set] =
            RG_MAX(binding_counts[binding->set], binding->binding + 1);

        vk_binding->binding = binding->binding;
        vk_binding->descriptorType = pipeline_binding_type_to_vk(binding->type);
        vk_binding->descriptorCount = 1;
        vk_binding->stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    pipeline->num_sets = num_sets;

    VkDescriptorSetLayout set_layouts[RG_MAX_DESCRIPTOR_SETS];
    for (uint32_t i = 0; i < pipeline->num_sets; ++i)
    {
        rgDescriptorPoolInit(device, &pipeline->pools[i], binding_counts[i], bindings[i]);
        set_layouts[i] = pipeline->pools[i].set_layout;
    }

    //
    // Create pipeline layout
    //

    VkPipelineLayoutCreateInfo pipeline_layout_info;
    memset(&pipeline_layout_info, 0, sizeof(pipeline_layout_info));
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = pipeline->num_sets;
    pipeline_layout_info.pSetLayouts = set_layouts;
    pipeline_layout_info.pushConstantRangeCount = 0;
    pipeline_layout_info.pPushConstantRanges = NULL;

    VK_CHECK(vkCreatePipelineLayout(
        device->device, &pipeline_layout_info, NULL, &pipeline->pipeline_layout));

    VkShaderModuleCreateInfo module_create_info;
    memset(&module_create_info, 0, sizeof(module_create_info));
    module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    module_create_info.codeSize = info->code_size;
    module_create_info.pCode = (uint32_t *)info->code;

    VK_CHECK(vkCreateShaderModule(
        device->device, &module_create_info, NULL, &pipeline->compute.shader));

    VkPipelineShaderStageCreateInfo stage_create_info;
    memset(&stage_create_info, 0, sizeof(stage_create_info));

    stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_create_info.module = pipeline->compute.shader;
    stage_create_info.pName = info->entry;

    VkComputePipelineCreateInfo pipeline_create_info;
    memset(&pipeline_create_info, 0, sizeof(pipeline_create_info));

    pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create_info.stage = stage_create_info;
    pipeline_create_info.layout = pipeline->pipeline_layout;

    vkCreateComputePipelines(
        device->device,
        VK_NULL_HANDLE,
        1,
        &pipeline_create_info,
        NULL,
        &pipeline->compute.instance);

    return pipeline;
}

void rgPipelineDestroy(RgDevice *device, RgPipeline *pipeline)
{
    VK_CHECK(vkDeviceWaitIdle(device->device));

    if (pipeline->bindings)
    {
        free(pipeline->bindings);
    }

    for (uint32_t i = 0; i < pipeline->num_sets; ++i)
    {
        rgDescriptorPoolDestroy(&pipeline->pools[i]);
    }

    vkDestroyPipelineLayout(device->device, pipeline->pipeline_layout, NULL);

    switch (pipeline->type)
    {
    case RG_PIPELINE_TYPE_GRAPHICS:
    {
        if (pipeline->graphics.vertex_entry)
        {
            free(pipeline->graphics.vertex_entry);
        }
        if (pipeline->graphics.fragment_entry)
        {
            free(pipeline->graphics.fragment_entry);
        }

        for (uint32_t i = 0; i < pipeline->graphics.instances.size; ++i)
        {
            if (pipeline->graphics.instances.hashes[i] != 0)
            {
                VkPipeline instance = VK_NULL_HANDLE;
                memcpy(
                    &instance,
                    &pipeline->graphics.instances.values[i],
                    sizeof(VkPipeline));
                assert(instance != VK_NULL_HANDLE);
                vkDestroyPipeline(device->device, instance, NULL);
            }
        }

        free(pipeline->graphics.vertex_attributes);

        if (pipeline->graphics.vertex_shader)
        {
            vkDestroyShaderModule(device->device, pipeline->graphics.vertex_shader, NULL);
        }

        if (pipeline->graphics.fragment_shader)
        {
            vkDestroyShaderModule(device->device, pipeline->graphics.fragment_shader, NULL);
        }

        rgHashmapDestroy(&pipeline->graphics.instances);
        break;
    }

    case RG_PIPELINE_TYPE_COMPUTE:
    {
        if (pipeline->compute.shader)
        {
            vkDestroyShaderModule(device->device, pipeline->compute.shader, NULL);
        }

        if (pipeline->compute.instance)
        {
            vkDestroyPipeline(device->device, pipeline->compute.instance, NULL);
        }
        break;
    }
    }

    free(pipeline);
}

static VkPipeline rgGraphicsPipelineGetInstance(
    RgDevice *device, RgPipeline *pipeline, RgPass *pass)
{
    uint64_t *found = rgHashmapGet(&pipeline->graphics.instances, pass->hash);
    if (found)
    {
        VkPipeline instance;
        memcpy(&instance, found, sizeof(VkPipeline));
        return instance;
    }

    uint32_t num_stages = 0;
    VkPipelineShaderStageCreateInfo stages[RG_MAX_SHADER_STAGES];
    memset(stages, 0, sizeof(stages));

    assert(pipeline->type == RG_PIPELINE_TYPE_GRAPHICS);

    if (pipeline->graphics.vertex_shader)
    {
        VkPipelineShaderStageCreateInfo *stage = &stages[num_stages++];
        stage->sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage->stage = VK_SHADER_STAGE_VERTEX_BIT;
        stage->module = pipeline->graphics.vertex_shader;
        stage->pName =
            pipeline->graphics.vertex_entry ?
            pipeline->graphics.vertex_entry : "main";
    }

    if (pipeline->graphics.fragment_shader)
    {
        VkPipelineShaderStageCreateInfo *stage = &stages[num_stages++];
        stage->sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage->stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stage->module = pipeline->graphics.fragment_shader;
        stage->pName =
            pipeline->graphics.fragment_entry ?
            pipeline->graphics.fragment_entry : "main";
    }

    VkPipelineVertexInputStateCreateInfo vertex_input_info;
    memset(&vertex_input_info, 0, sizeof(vertex_input_info));
    vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkVertexInputBindingDescription vertex_binding;
    memset(&vertex_binding, 0, sizeof(vertex_binding));

    VkVertexInputAttributeDescription attributes[RG_MAX_VERTEX_ATTRIBUTES];
    memset(attributes, 0, sizeof(attributes));

    if (pipeline->graphics.vertex_stride > 0)
    {
        assert(pipeline->graphics.num_vertex_attributes > 0);

        vertex_binding.binding = 0;
        vertex_binding.stride = pipeline->graphics.vertex_stride;

        vertex_input_info.vertexBindingDescriptionCount = 1;
        vertex_input_info.pVertexBindingDescriptions = &vertex_binding;

        for (uint32_t i = 0; i < pipeline->graphics.num_vertex_attributes; ++i)
        {
            attributes[i].binding = 0;
            attributes[i].location = i;
            attributes[i].format =
                format_to_vk(pipeline->graphics.vertex_attributes[i].format);
            attributes[i].offset = pipeline->graphics.vertex_attributes[i].offset;
        }

        vertex_input_info.vertexAttributeDescriptionCount =
            pipeline->graphics.num_vertex_attributes;
        vertex_input_info.pVertexAttributeDescriptions = attributes;
    }

    VkPipelineInputAssemblyStateCreateInfo input_assembly;
    memset(&input_assembly, 0, sizeof(input_assembly));
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = primitive_topology_to_vk(pipeline->graphics.topology);
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport;
    memset(&viewport, 0, sizeof(viewport));
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)pass->extent.width;
    viewport.height = (float)pass->extent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor;
    memset(&scissor, 0, sizeof(scissor));
    scissor.offset = (VkOffset2D){0, 0};
    scissor.extent = pass->extent;

    VkPipelineViewportStateCreateInfo viewport_state;
    memset(&viewport_state, 0, sizeof(viewport_state));
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizer;
    memset(&rasterizer, 0, sizeof(rasterizer));
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = polygon_mode_to_vk(pipeline->graphics.polygon_mode);
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = cull_mode_to_vk(pipeline->graphics.cull_mode);
    rasterizer.frontFace = front_face_to_vk(pipeline->graphics.front_face);
    rasterizer.depthBiasEnable = pipeline->graphics.depth_stencil.bias_enable;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampling;
    memset(&multisampling, 0, sizeof(multisampling));
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 0.0f;          // Optional
    multisampling.pSampleMask = NULL;               // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE;      // Optional

    VkPipelineDepthStencilStateCreateInfo depth_stencil;
    memset(&depth_stencil, 0, sizeof(depth_stencil));
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = pipeline->graphics.depth_stencil.test_enable;
    depth_stencil.depthWriteEnable = pipeline->graphics.depth_stencil.write_enable;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState color_blend_attachment_enabled;
    memset(&color_blend_attachment_enabled, 0, sizeof(color_blend_attachment_enabled));
    color_blend_attachment_enabled.blendEnable = VK_TRUE;
    color_blend_attachment_enabled.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment_enabled.dstColorBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment_enabled.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment_enabled.srcAlphaBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment_enabled.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment_enabled.alphaBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment_enabled.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendAttachmentState color_blend_attachment_disabled;
    memset(&color_blend_attachment_disabled, 0, sizeof(color_blend_attachment_disabled));
    color_blend_attachment_disabled.blendEnable = VK_FALSE;
    color_blend_attachment_disabled.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    color_blend_attachment_disabled.dstColorBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment_disabled.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment_disabled.srcAlphaBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    color_blend_attachment_disabled.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment_disabled.alphaBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment_disabled.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendAttachmentState blend_infos[RG_MAX_COLOR_ATTACHMENTS];
    assert(pass->num_color_attachments <= RG_LENGTH(blend_infos));

    if (pipeline->graphics.blend.enable)
    {
        for (uint32_t i = 0; i < RG_LENGTH(blend_infos); ++i)
        {
            blend_infos[i] = color_blend_attachment_enabled;
        }
    }
    else
    {
        for (uint32_t i = 0; i < RG_LENGTH(blend_infos); ++i)
        {
            blend_infos[i] = color_blend_attachment_disabled;
        }
    }

    VkPipelineColorBlendStateCreateInfo color_blending;
    memset(&color_blending, 0, sizeof(color_blending));
    color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY; // Optional
    color_blending.attachmentCount = pass->num_color_attachments;
    color_blending.pAttachments = blend_infos;

    VkDynamicState dynamic_states[3] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
    };

    VkPipelineDynamicStateCreateInfo dynamic_state;
    memset(&dynamic_state, 0, sizeof(dynamic_state));
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = RG_LENGTH(dynamic_states);
    dynamic_state.pDynamicStates = dynamic_states;

    VkGraphicsPipelineCreateInfo pipeline_info;
    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount = num_stages;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = pipeline->pipeline_layout;
    pipeline_info.renderPass = pass->renderpass;
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_info.basePipelineIndex = -1;

    VkPipeline instance = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(
        device->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &instance));

    uint64_t instance_id = 0;
    memcpy(&instance_id, &instance, sizeof(VkPipeline));
    rgHashmapSet(&pipeline->graphics.instances, pass->hash, instance_id);

    return instance;
}
// }}}

// Command buffer {{{
static RgCmdBuffer *allocateCmdBuffer(RgDevice *device, RgCmdPool *cmd_pool)
{
    RgCmdBuffer *cmd_buffer = malloc(sizeof(*cmd_buffer));
    memset(cmd_buffer, 0, sizeof(*cmd_buffer));

    cmd_buffer->device = device;
    cmd_buffer->cmd_pool = cmd_pool;

    VkCommandBufferAllocateInfo alloc_info;
    memset(&alloc_info, 0, sizeof(alloc_info));
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = cmd_pool->command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VK_CHECK(
        vkAllocateCommandBuffers(device->device, &alloc_info, &cmd_buffer->cmd_buffer));

    rgBufferPoolInit(
        device,
        &cmd_buffer->ubo_pool,
        RG_BUFFER_POOL_CHUNK_SIZE, /*chunk size*/
        RG_MAX(
            16,
            device->physical_device_properties.limits
                .minUniformBufferOffsetAlignment), /*alignment*/
        RG_BUFFER_USAGE_UNIFORM);

    rgBufferPoolInit(
        device,
        &cmd_buffer->vbo_pool,
        RG_BUFFER_POOL_CHUNK_SIZE, /*chunk size*/
        16,                        /*alignment*/
        RG_BUFFER_USAGE_VERTEX);

    rgBufferPoolInit(
        device,
        &cmd_buffer->ibo_pool,
        RG_BUFFER_POOL_CHUNK_SIZE, /*chunk size*/
        16,                        /*alignment*/
        RG_BUFFER_USAGE_INDEX);

    return cmd_buffer;
}

static void freeCmdBuffer(RgDevice *device, RgCmdBuffer *cmd_buffer)
{
    rgBufferPoolDestroy(&cmd_buffer->ubo_pool);
    rgBufferPoolDestroy(&cmd_buffer->vbo_pool);
    rgBufferPoolDestroy(&cmd_buffer->ibo_pool);

    vkFreeCommandBuffers(
        device->device, cmd_buffer->cmd_pool->command_pool, 1, &cmd_buffer->cmd_buffer);

    free(cmd_buffer);
}

static void cmdBufferBindDescriptors(RgCmdBuffer *cmd_buffer)
{
    assert(cmd_buffer->current_pipeline);

    for (uint32_t i = 0; i < cmd_buffer->current_pipeline->num_sets; ++i)
    {
        RgDescriptorPool *pool = &cmd_buffer->current_pipeline->pools[i];

        RgDescriptor *descriptors = &cmd_buffer->bound_descriptors[i][0];

        VkDescriptorSet descriptor_set =
            rgDescriptorPoolAllocate(pool, pool->num_bindings, descriptors);


        VkPipelineBindPoint bind_point;
        switch (cmd_buffer->current_pipeline->type)
        {
        case RG_PIPELINE_TYPE_GRAPHICS: bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS; break;
        case RG_PIPELINE_TYPE_COMPUTE: bind_point = VK_PIPELINE_BIND_POINT_COMPUTE; break;
        }

        vkCmdBindDescriptorSets(
            cmd_buffer->cmd_buffer,
            bind_point,
            cmd_buffer->current_pipeline->pipeline_layout,
            i,
            1,
            &descriptor_set,
            0,
            NULL);
    }
}

void rgCmdSetViewport(RgCmdBuffer *cb, const RgViewport *viewport)
{
    VkViewport vk_viewport;
    vk_viewport.width = viewport->width;
    vk_viewport.height = viewport->height;
    vk_viewport.x = viewport->x;
    vk_viewport.y = viewport->y;
    vk_viewport.minDepth = viewport->min_depth;
    vk_viewport.maxDepth = viewport->max_depth;

    vkCmdSetViewport(cb->cmd_buffer, 0, 1, &vk_viewport);
}

void rgCmdSetScissor(RgCmdBuffer *cb, const RgRect2D *rect)
{
    VkRect2D vk_scissor;
    vk_scissor.offset.x = rect->offset.x;
    vk_scissor.offset.y = rect->offset.y;
    vk_scissor.extent.width = rect->extent.width;
    vk_scissor.extent.height = rect->extent.height;
    vkCmdSetScissor(cb->cmd_buffer, 0, 1, &vk_scissor);
}

void rgCmdBindPipeline(RgCmdBuffer *cmd_buffer, RgPipeline *pipeline)
{
    cmd_buffer->current_pipeline = pipeline;

    switch (pipeline->type)
    {
    case RG_PIPELINE_TYPE_GRAPHICS:
    {
        RgPass *pass = cmd_buffer->current_pass;
        assert(pass);

        VkPipeline instance = rgGraphicsPipelineGetInstance(
            cmd_buffer->device, pipeline, pass);

        assert(instance != VK_NULL_HANDLE);

        vkCmdBindPipeline(
            cmd_buffer->cmd_buffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            instance);
        break;
    }
    case RG_PIPELINE_TYPE_COMPUTE:
    {
        assert(pipeline->compute.instance != VK_NULL_HANDLE);

        vkCmdBindPipeline(
            cmd_buffer->cmd_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline->compute.instance);
        break;
    }
    }
}

void rgCmdBindImage(
    RgCmdBuffer *cmd_buffer, uint32_t binding, uint32_t set, RgImage *image)
{
    RgDescriptor descriptor;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.image.imageView = image->view;
    descriptor.image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    cmd_buffer->bound_descriptors[set][binding] = descriptor;
}

void rgCmdBindSampler(
    RgCmdBuffer *cmd_buffer, uint32_t binding, uint32_t set, RgSampler *sampler)
{
    RgDescriptor descriptor;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.image.sampler = sampler->sampler;

    cmd_buffer->bound_descriptors[set][binding] = descriptor;
}

void rgCmdBindImageSampler(
    RgCmdBuffer *cmd_buffer,
    uint32_t binding,
    uint32_t set,
    RgImage *image,
    RgSampler *sampler)
{
    RgDescriptor descriptor;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.image.imageView = image->view;
    descriptor.image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    descriptor.image.sampler = sampler->sampler;

    cmd_buffer->bound_descriptors[set][binding] = descriptor;
}

void rgCmdSetUniform(
    RgCmdBuffer *cmd_buffer, uint32_t binding, uint32_t set, size_t size, void *data)
{
    RgBufferAllocation alloc = rgBufferPoolAllocate(&cmd_buffer->ubo_pool, size);
    memcpy(alloc.mapping, data, size);

    RgDescriptor descriptor;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.buffer.buffer = alloc.buffer->buffer;
    descriptor.buffer.offset = alloc.offset;
    descriptor.buffer.range = alloc.size;

    cmd_buffer->bound_descriptors[set][binding] = descriptor;
}

void rgCmdSetVertices(RgCmdBuffer *cmd_buffer, size_t size, void *data)
{
    RgBufferAllocation alloc = rgBufferPoolAllocate(&cmd_buffer->vbo_pool, size);
    memcpy(alloc.mapping, data, size);

    vkCmdBindVertexBuffers(
        cmd_buffer->cmd_buffer, 0, 1, &alloc.buffer->buffer, &alloc.offset);
}

void rgCmdSetIndices(
    RgCmdBuffer *cmd_buffer, RgIndexType index_type, size_t size, void *data)
{
    RgBufferAllocation alloc = rgBufferPoolAllocate(&cmd_buffer->ibo_pool, size);
    memcpy(alloc.mapping, data, size);

    vkCmdBindIndexBuffer(
        cmd_buffer->cmd_buffer,
        alloc.buffer->buffer,
        alloc.offset,
        index_type_to_vk(index_type));
}

void rgCmdBindVertexBuffer(RgCmdBuffer *cmd_buffer, RgBuffer *buffer, size_t offset)
{
    vkCmdBindVertexBuffers(cmd_buffer->cmd_buffer, 0, 1, &buffer->buffer, &offset);
}

void rgCmdBindIndexBuffer(
    RgCmdBuffer *cmd_buffer, RgIndexType index_type, RgBuffer *buffer, size_t offset)
{
    vkCmdBindIndexBuffer(
        cmd_buffer->cmd_buffer, buffer->buffer, offset, index_type_to_vk(index_type));
}

void rgCmdBindUniformBuffer(
    RgCmdBuffer *cb,
    uint32_t binding,
    uint32_t set,
    RgBuffer *buffer,
    size_t offset,
    size_t range)
{
    RgDescriptor descriptor;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.buffer.buffer = buffer->buffer;
    descriptor.buffer.offset = offset;
    descriptor.buffer.range = (range != 0) ? range : VK_WHOLE_SIZE;

    cb->bound_descriptors[set][binding] = descriptor;
}

void rgCmdBindStorageBuffer(
    RgCmdBuffer *cb,
    uint32_t binding,
    uint32_t set,
    RgBuffer *buffer,
    size_t offset,
    size_t range)
{
    RgDescriptor descriptor;
    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.buffer.buffer = buffer->buffer;
    descriptor.buffer.offset = offset;
    descriptor.buffer.range = (range != 0) ? range : VK_WHOLE_SIZE;

    cb->bound_descriptors[set][binding] = descriptor;
}

void rgCmdDraw(
    RgCmdBuffer *cmd_buffer,
    uint32_t vertex_count,
    uint32_t instance_count,
    uint32_t first_vertex,
    uint32_t first_instance)
{
    cmdBufferBindDescriptors(cmd_buffer);

    vkCmdDraw(
        cmd_buffer->cmd_buffer,
        vertex_count,
        instance_count,
        first_vertex,
        first_instance);
}

void rgCmdDrawIndexed(
    RgCmdBuffer *cmd_buffer,
    uint32_t index_count,
    uint32_t instance_count,
    uint32_t first_index,
    int32_t vertex_offset,
    uint32_t first_instance)
{
    cmdBufferBindDescriptors(cmd_buffer);

    vkCmdDrawIndexed(
        cmd_buffer->cmd_buffer,
        index_count,
        instance_count,
        first_index,
        vertex_offset,
        first_instance);
}

void rgCmdDispatch(
    RgCmdBuffer *cmd_buffer,
    uint32_t group_count_x,
    uint32_t group_count_y,
    uint32_t group_count_z)
{
    cmdBufferBindDescriptors(cmd_buffer);

    vkCmdDispatch(
        cmd_buffer->cmd_buffer,
        group_count_x,
        group_count_y,
        group_count_z);
}

void rgCmdCopyBufferToBuffer(
    RgCmdBuffer *cmd_buffer,
    RgBuffer *src,
    size_t src_offset,
    RgBuffer *dst,
    size_t dst_offset,
    size_t size)
{
    VkBufferCopy region;
    memset(&region, 0, sizeof(region));
    region.srcOffset = src_offset;
    region.dstOffset = dst_offset;
    region.size = size;
    vkCmdCopyBuffer(cmd_buffer->cmd_buffer, src->buffer, dst->buffer, 1, &region);
}

void rgCmdCopyBufferToImage(
    RgCmdBuffer *cmd_buffer, RgBufferCopy *src, RgImageCopy *dst, RgExtent3D extent)
{
    VkImageSubresourceLayers subresource;
    memset(&subresource, 0, sizeof(subresource));
    subresource.aspectMask = dst->image->aspect;
    subresource.mipLevel = dst->mip_level;
    subresource.baseArrayLayer = dst->array_layer;
    subresource.layerCount = 1;

    VkBufferImageCopy region = {
        .bufferOffset = src->offset,
        .bufferRowLength = src->row_length,
        .bufferImageHeight = src->image_height,
        .imageSubresource = subresource,
        .imageOffset =
            (VkOffset3D){
                .x = dst->offset.x,
                .y = dst->offset.y,
                .z = dst->offset.z,
            },
        .imageExtent =
            (VkExtent3D){
                .width = extent.width,
                .height = extent.height,
                .depth = extent.depth,
            },
    };

    vkCmdCopyBufferToImage(
        cmd_buffer->cmd_buffer,
        src->buffer->buffer,
        dst->image->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region);
}

void rgCmdCopyImageToBuffer(
    RgCmdBuffer *cmd_buffer, RgImageCopy *src, RgBufferCopy *dst, RgExtent3D extent)
{
    VkImageSubresourceLayers subresource;
    memset(&subresource, 0, sizeof(subresource));
    subresource.aspectMask = src->image->aspect;
    subresource.mipLevel = src->mip_level;
    subresource.baseArrayLayer = src->array_layer;
    subresource.layerCount = 1;

    VkBufferImageCopy region = {
        .bufferOffset = dst->offset,
        .bufferRowLength = dst->row_length,
        .bufferImageHeight = dst->image_height,
        .imageSubresource = subresource,
        .imageOffset =
            (VkOffset3D){
                .x = src->offset.x,
                .y = src->offset.y,
                .z = src->offset.z,
            },
        .imageExtent =
            (VkExtent3D){
                .width = extent.width,
                .height = extent.height,
                .depth = extent.depth,
            },
    };

    vkCmdCopyImageToBuffer(
        cmd_buffer->cmd_buffer,
        src->image->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst->buffer->buffer,
        1,
        &region);
}

void rgCmdCopyImageToImage(
    RgCmdBuffer *cmd_buffer, RgImageCopy *src, RgImageCopy *dst, RgExtent3D extent)
{
    VkImageSubresourceLayers src_subresource;
    memset(&src_subresource, 0, sizeof(src_subresource));
    src_subresource.aspectMask = src->image->aspect;
    src_subresource.mipLevel = src->mip_level;
    src_subresource.baseArrayLayer = src->array_layer;
    src_subresource.layerCount = 1;

    VkImageSubresourceLayers dst_subresource;
    memset(&dst_subresource, 0, sizeof(dst_subresource));
    dst_subresource.aspectMask = dst->image->aspect;
    dst_subresource.mipLevel = dst->mip_level;
    dst_subresource.baseArrayLayer = dst->array_layer;
    dst_subresource.layerCount = 1;

    VkImageCopy region = {
        .srcSubresource = src_subresource,
        .srcOffset = {.x = src->offset.x, .y = src->offset.y, .z = src->offset.z},
        .dstSubresource = dst_subresource,
        .dstOffset = {.x = dst->offset.x, .y = dst->offset.y, .z = dst->offset.z},
        .extent = {.width = extent.width, .height = extent.height, .depth = extent.depth},
    };

    vkCmdCopyImage(
        cmd_buffer->cmd_buffer,
        src->image->image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        dst->image->image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region);
}
// }}}

// Graph {{{
static void rgResourceResolveImageInfo(
    const RgGraph *graph,
    const RgGraphImageInfoInternal *internal,
    RgImageInfo *out_info)
{
    memset(out_info, 0, sizeof(*out_info));

    out_info->depth = internal->depth;
    out_info->sample_count = internal->sample_count;
    out_info->mip_count = internal->mip_count;
    out_info->layer_count = internal->layer_count;
    out_info->usage = internal->usage;
    out_info->aspect = internal->aspect;
    out_info->format = internal->format;

    switch (internal->scaling_mode)
    {
    case RG_GRAPH_IMAGE_SCALING_MODE_ABSOLUTE: {
        out_info->width = (uint32_t)internal->width;
        out_info->height = (uint32_t)internal->height;
        break;
    }
    case RG_GRAPH_IMAGE_SCALING_MODE_RELATIVE: {
        assert(graph->has_swapchain);
        assert(internal->width <= 1.0);
        assert(internal->height <= 1.0);

        assert(graph->has_swapchain);
        out_info->width = (uint32_t)(
            internal->width * (float)graph->swapchain.extent.width);
        out_info->height = (uint32_t)(
            internal->height * (float)graph->swapchain.extent.height);
        break;
    }
    }
}

static void
rgNodeInit(RgGraph *graph, RgNode *node, uint32_t *pass_indices, uint32_t pass_count)
{
    (void)graph;
    memset(node, 0, sizeof(*node));

    node->num_pass_indices = pass_count;
    node->pass_indices = (uint32_t *)malloc(sizeof(*node->pass_indices) * pass_count);
    memcpy(node->pass_indices, pass_indices, sizeof(*node->pass_indices) * pass_count);
}

static void rgNodeDestroy(RgGraph *graph, RgNode *node)
{
    RgDevice *device = graph->device;
    VK_CHECK(vkDeviceWaitIdle(device->device));

    assert(graph->num_frames > 0);
    for (uint32_t i = 0; i < graph->num_frames; ++i)
    {
        vkDestroySemaphore(
            device->device, node->frames[i].execution_finished_semaphore, NULL);
        node->frames[i].execution_finished_semaphore = VK_NULL_HANDLE;

        vkDestroyFence(device->device, node->frames[i].fence, NULL);
        node->frames[i].fence = VK_NULL_HANDLE;

        freeCmdBuffer(device, node->frames[i].cmd_buffer);
        node->frames[i].cmd_buffer = NULL;

        arrFree(&node->frames[i].wait_semaphores);
        arrFree(&node->frames[i].wait_stages);
    }

    free(node->pass_indices);
}

static void
rgNodeResize(RgGraph *graph, RgNode *node)
{
    VK_CHECK(vkDeviceWaitIdle(graph->device->device));

    bool is_last = node == &graph->nodes.ptr[graph->nodes.len - 1];

    for (uint32_t i = 0; i < graph->num_frames; ++i)
    {
        if (node->frames[i].cmd_buffer != NULL)
        {
            freeCmdBuffer(graph->device, node->frames[i].cmd_buffer);
            node->frames[i].cmd_buffer = NULL;
        }

        if (node->frames[i].execution_finished_semaphore != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(
                graph->device->device, node->frames[i].execution_finished_semaphore, NULL);
            node->frames[i].execution_finished_semaphore = VK_NULL_HANDLE;
        }

        if (node->frames[i].fence != VK_NULL_HANDLE)
        {
            vkDestroyFence(
                graph->device->device, node->frames[i].fence, NULL);
            node->frames[i].fence = VK_NULL_HANDLE;
        }

        node->frames[i].cmd_buffer = allocateCmdBuffer(graph->device, graph->cmd_pool);

        VkSemaphoreCreateInfo semaphore_info;
        memset(&semaphore_info, 0, sizeof(semaphore_info));
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(
            graph->device->device,
            &semaphore_info,
            NULL,
            &node->frames[i].execution_finished_semaphore));

        VkFenceCreateInfo fence_info;
        memset(&fence_info, 0, sizeof(fence_info));
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK(vkCreateFence(
            graph->device->device, &fence_info, NULL, &node->frames[i].fence));

        arrFree(&node->frames[i].wait_semaphores);
        arrFree(&node->frames[i].wait_stages);

        memset(&node->frames[i].wait_semaphores, 0, sizeof(node->frames[i].wait_semaphores));
        memset(&node->frames[i].wait_stages, 0, sizeof(node->frames[i].wait_stages));

        if (is_last && graph->has_swapchain)
        {
            arrPush(&node->frames[i].wait_semaphores, graph->image_available_semaphores[i]);
            arrPush(&node->frames[i].wait_stages, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        }
    }
}

static uint64_t rgRenderpassHash(VkRenderPassCreateInfo *ci)
{
    uint64_t hash = 0;
    fnvHashReset(&hash);

    fnvHashUpdate(
        &hash,
        (uint8_t *)ci->pAttachments,
        ci->attachmentCount * sizeof(*ci->pAttachments));

    for (uint32_t i = 0; i < ci->subpassCount; i++)
    {
        const VkSubpassDescription *subpass = &ci->pSubpasses[i];
        fnvHashUpdate(
            &hash,
            (uint8_t *)&subpass->pipelineBindPoint,
            sizeof(subpass->pipelineBindPoint));

        fnvHashUpdate(&hash, (uint8_t *)&subpass->flags, sizeof(subpass->flags));

        if (subpass->pColorAttachments)
        {
            fnvHashUpdate(
                &hash,
                (uint8_t *)subpass->pColorAttachments,
                subpass->colorAttachmentCount * sizeof(*subpass->pColorAttachments));
        }

        if (subpass->pResolveAttachments)
        {
            fnvHashUpdate(
                &hash,
                (uint8_t *)subpass->pResolveAttachments,
                subpass->colorAttachmentCount * sizeof(*subpass->pResolveAttachments));
        }

        if (subpass->pDepthStencilAttachment)
        {
            fnvHashUpdate(
                &hash,
                (uint8_t *)subpass->pDepthStencilAttachment,
                sizeof(*subpass->pDepthStencilAttachment));
        }

        if (subpass->pInputAttachments)
        {
            fnvHashUpdate(
                &hash,
                (uint8_t *)subpass->pInputAttachments,
                subpass->inputAttachmentCount * sizeof(*subpass->pInputAttachments));
        }

        if (subpass->pPreserveAttachments)
        {
            fnvHashUpdate(
                &hash,
                (uint8_t *)subpass->pPreserveAttachments,
                subpass->preserveAttachmentCount * sizeof(*subpass->pPreserveAttachments));
        }
    }

    fnvHashUpdate(
        &hash,
        (uint8_t *)ci->pDependencies,
        ci->dependencyCount * sizeof(*ci->pDependencies));

    return hash;
}

static void rgPassResize(RgGraph *graph, RgPass *pass)
{
    if (pass->type != RG_PASS_TYPE_GRAPHICS) return; 

    assert(graph->num_frames > 0);
    for (uint32_t i = 0; i < graph->num_frames; ++i)
    {
        for (uint32_t j = 0; j < pass->num_framebuffers; ++j)
        {
            if (pass->frames[i].framebuffers[j])
            {
                vkDestroyFramebuffer(graph->device->device,
                                     pass->frames[i].framebuffers[j], NULL);
                pass->frames[i].framebuffers[j] = VK_NULL_HANDLE;
            }
        }

        if (pass->frames[i].framebuffers)
        {
            free(pass->frames[i].framebuffers);
        }
    }

    if (pass->renderpass)
    {
        vkDestroyRenderPass(graph->device->device, pass->renderpass, NULL);
        pass->renderpass = VK_NULL_HANDLE;
    }

    if (pass->is_backbuffer)
    {
        assert(graph->has_swapchain);
        pass->extent = graph->swapchain.extent;
    }
    else
    {
        for (uint32_t i = 0; i < pass->used_resources.len; ++i)
        {
            RgPassResource pass_res = pass->used_resources.ptr[i];
            RgResource *resource = &graph->resources.ptr[pass_res.index];

            switch (resource->type)
            {
            case RG_RESOURCE_IMAGE:
            case RG_RESOURCE_EXTERNAL_IMAGE:
            {
                switch (pass_res.post_usage)
                {
                case RG_RESOURCE_USAGE_COLOR_ATTACHMENT:
                case RG_RESOURCE_USAGE_DEPTH_STENCIL_ATTACHMENT:
                {
                    RgImageInfo image_info;
                    rgResourceResolveImageInfo(graph, &resource->image_info, &image_info);
                    pass->extent.width = image_info.width;
                    pass->extent.height = image_info.height;
                    break;
                }

                default: break;
                }
                break;
            }

            default: break;
            }
        }
    }

    if (pass->is_backbuffer)
    {
        pass->num_framebuffers = graph->swapchain.num_images;
    }
    else
    {
        pass->num_framebuffers = 1;
    }

    assert(graph->num_frames > 0);
    for (uint32_t i = 0; i < graph->num_frames; ++i)
    {
        pass->frames[i].framebuffers =
            (VkFramebuffer *)malloc(sizeof(VkFramebuffer) * pass->num_framebuffers);
        memset(pass->frames[i].framebuffers,
               0, sizeof(VkFramebuffer) * pass->num_framebuffers);
    }

    ARRAY_OF(VkAttachmentDescription) rp_attachments;
    memset(&rp_attachments, 0, sizeof(rp_attachments));

    ARRAY_OF(VkAttachmentReference) color_attachment_refs;
    memset(&color_attachment_refs, 0, sizeof(color_attachment_refs));

    bool has_depth_stencil = false;
    VkAttachmentReference depth_stencil_attachment_ref;
    memset(&depth_stencil_attachment_ref, 0, sizeof(depth_stencil_attachment_ref));

    if (pass->is_backbuffer)
    {
        VkAttachmentDescription backbuffer;
        memset(&backbuffer, 0, sizeof(backbuffer));
        backbuffer.format = graph->swapchain.image_format;
        backbuffer.samples = VK_SAMPLE_COUNT_1_BIT;
        backbuffer.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        backbuffer.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        backbuffer.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        backbuffer.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        backbuffer.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        backbuffer.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        arrPush(&rp_attachments, backbuffer);

        VkAttachmentReference backbuffer_ref;
        memset(&backbuffer_ref, 0, sizeof(backbuffer_ref));
        backbuffer_ref.attachment = rp_attachments.len - 1;
        backbuffer_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        arrPush(&color_attachment_refs, backbuffer_ref);
    }

    for (uint32_t i = 0; i < pass->used_resources.len; ++i)
    {
        RgPassResource pass_res = pass->used_resources.ptr[i];
        RgResource *resource = &graph->resources.ptr[pass_res.index];

        switch (pass_res.post_usage)
        {
        case RG_RESOURCE_USAGE_COLOR_ATTACHMENT: {
            VkAttachmentDescription attachment;
            memset(&attachment, 0, sizeof(attachment));
            attachment.format = format_to_vk(resource->image_info.format);
            attachment.samples = (VkSampleCountFlagBits)resource->image_info.sample_count;
            attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            arrPush(&rp_attachments, attachment);

            VkAttachmentReference attachment_ref;
            memset(&attachment_ref, 0, sizeof(attachment_ref));
            attachment_ref.attachment = rp_attachments.len - 1;
            attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            arrPush(&color_attachment_refs, attachment_ref);
            break;
        }

        case RG_RESOURCE_USAGE_DEPTH_STENCIL_ATTACHMENT: {
            has_depth_stencil = true;

            VkAttachmentDescription attachment;
            memset(&attachment, 0, sizeof(attachment));
            attachment.format = format_to_vk(resource->image_info.format);
            attachment.samples = VK_SAMPLE_COUNT_1_BIT;
            attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            arrPush(&rp_attachments, attachment);

            pass->depth_attachment_index = rp_attachments.len - 1;
            depth_stencil_attachment_ref.attachment = rp_attachments.len - 1;
            depth_stencil_attachment_ref.layout =
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            break;
        }

        default: break;
        }
    }

    VkSubpassDescription subpass;
    memset(&subpass, 0, sizeof(subpass));
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = color_attachment_refs.len;
    subpass.pColorAttachments = color_attachment_refs.ptr;

    if (has_depth_stencil)
    {
        subpass.pDepthStencilAttachment = &depth_stencil_attachment_ref;
    }

    RgNode *node = &graph->nodes.ptr[pass->node_index];
    uint32_t last_pass_in_node_index = node->pass_indices[node->num_pass_indices-1];
    RgPass *last_pass_in_node = &graph->passes.ptr[last_pass_in_node_index];
    bool is_last_pass_in_node = last_pass_in_node == pass;

    VkSubpassDependency dependencies[2];
    memset(dependencies, 0, sizeof(dependencies));

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    if (is_last_pass_in_node)
    {
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
            | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT
            | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    }
    else
    {
        // If this renderpass is not the last one, it means that its results
        // are going to be read from a subsequent renderpass
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    }

    VkRenderPassCreateInfo renderpass_ci;
    memset(&renderpass_ci, 0, sizeof(renderpass_ci));
    renderpass_ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderpass_ci.attachmentCount = rp_attachments.len;
    renderpass_ci.pAttachments = rp_attachments.ptr;
    renderpass_ci.subpassCount = 1;
    renderpass_ci.pSubpasses = &subpass;
    renderpass_ci.dependencyCount = sizeof(dependencies) / sizeof(dependencies[0]);
    renderpass_ci.pDependencies = dependencies;

    VK_CHECK(vkCreateRenderPass(
        graph->device->device, &renderpass_ci, NULL, &pass->renderpass));

    pass->hash = rgRenderpassHash(&renderpass_ci);

    assert(graph->num_frames > 0);
    for (uint32_t f = 0; f < graph->num_frames; ++f)
    {
        for (uint32_t i = 0; i < pass->num_framebuffers; ++i)
        {
            ARRAY_OF(VkImageView) views;
            memset(&views, 0, sizeof(views));

            if (pass->is_backbuffer)
            {
                arrPush(&views, graph->swapchain.image_views[i]);
            }

            for (uint32_t j = 0; j < pass->used_resources.len; ++j)
            {
                RgPassResource pass_res = pass->used_resources.ptr[j];
                RgResource *resource = &graph->resources.ptr[pass_res.index];

                switch (pass_res.post_usage)
                {
                case RG_RESOURCE_USAGE_COLOR_ATTACHMENT:
                case RG_RESOURCE_USAGE_DEPTH_STENCIL_ATTACHMENT:
                {
                    arrPush(&views, resource->frames[f].image->view);

                    assert(pass->extent.width == resource->frames[f].image->info.width);
                    assert(pass->extent.height == resource->frames[f].image->info.height);
                    break;
                }
                default: break;
                }
            }

            assert(views.len > 0);
            assert(pass->extent.width > 0);
            assert(pass->extent.height > 0);

            VkFramebufferCreateInfo create_info;
            memset(&create_info, 0, sizeof(create_info));
            create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            create_info.renderPass = pass->renderpass;
            create_info.attachmentCount = views.len;
            create_info.pAttachments = views.ptr;
            create_info.width = pass->extent.width;
            create_info.height = pass->extent.height;
            create_info.layers = 1;

            assert(create_info.attachmentCount == renderpass_ci.attachmentCount);

            VK_CHECK(vkCreateFramebuffer(
                graph->device->device, &create_info, NULL,
                &pass->frames[f].framebuffers[i]));

            arrFree(&views);
        }
    }

    arrFree(&rp_attachments);
    arrFree(&color_attachment_refs);
}

static void rgPassBuild(RgGraph *graph, RgPass *pass)
{
    (void)graph;
    if (pass->is_backbuffer)
    {
        pass->num_color_attachments++;
        pass->num_attachments++;
    }

    for (uint32_t i = 0; i < pass->used_resources.len; ++i)
    {
        RgPassResource pass_res = pass->used_resources.ptr[i];
        switch (pass_res.post_usage)
        {
        case RG_RESOURCE_USAGE_COLOR_ATTACHMENT:
            pass->num_color_attachments++;
            pass->num_attachments++;
            break;
        case RG_RESOURCE_USAGE_DEPTH_STENCIL_ATTACHMENT:
            pass->has_depth_attachment = true;
            pass->num_attachments++;
            break;
        default: break;
        }
    }

    pass->clear_values = malloc(sizeof(VkClearValue) * pass->num_attachments);
    memset(pass->clear_values, 0, sizeof(VkClearValue) * pass->num_attachments);
}

static void rgPassDestroy(RgGraph *graph, RgPass *pass)
{
    RgDevice *device = graph->device;

    VK_CHECK(vkDeviceWaitIdle(device->device));
    if (pass->renderpass)
    {
        vkDestroyRenderPass(device->device, pass->renderpass, NULL);
        pass->renderpass = VK_NULL_HANDLE;
    }

    assert(graph->num_frames > 0);
    for (uint32_t i = 0; i < graph->num_frames; ++i)
    {
        for (uint32_t j = 0; j < pass->num_framebuffers; ++j)
        {
            vkDestroyFramebuffer(device->device, pass->frames[i].framebuffers[j], NULL);
            if (pass->frames[i].framebuffers[j])
            {
                pass->frames[i].framebuffers[j] = VK_NULL_HANDLE;
            }
        }

        if (pass->frames[i].framebuffers)
        {
            free(pass->frames[i].framebuffers);
        }
    }

    arrFree(&pass->used_resources);
    free(pass->clear_values);
}

static void rgResourceResize(RgGraph *graph, RgResource* resource)
{
    assert(graph->num_frames > 0);
    for (uint32_t i = 0; i < graph->num_frames; ++i)
    {
        switch (resource->type)
        {
        case RG_RESOURCE_IMAGE: {
            if (resource->frames[i].image)
            {
                rgImageDestroy(graph->device, resource->frames[i].image);
            }

            RgImageInfo image_info;
            rgResourceResolveImageInfo(graph, &resource->image_info, &image_info);

            resource->frames[i].image = rgImageCreate(graph->device, &image_info);
            break;
        }

        case RG_RESOURCE_BUFFER:
            if (resource->frames[i].buffer) break;
            resource->frames[i].buffer =
                rgBufferCreate(graph->device, &resource->buffer_info);
            break;

        case RG_RESOURCE_EXTERNAL_BUFFER:
        case RG_RESOURCE_EXTERNAL_IMAGE: break;
        }
    }
}

void rgGraphResize(RgGraph *graph, uint32_t width, uint32_t height)
{
    VK_CHECK(vkDeviceWaitIdle(graph->device->device));

    for (uint32_t i = 0; i < graph->num_frames; ++i)
    {
        if (graph->image_available_semaphores[i] != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(
                graph->device->device, graph->image_available_semaphores[i], NULL);
        }

        VkSemaphoreCreateInfo semaphore_info;
        memset(&semaphore_info, 0, sizeof(semaphore_info));
        semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK(vkCreateSemaphore(
            graph->device->device,
            &semaphore_info,
            NULL,
            &graph->image_available_semaphores[i]));
    }

    if (graph->has_swapchain)
    {
        rgSwapchainResize(&graph->swapchain, width, height);
    }

    for (uint32_t i = 0; i < graph->nodes.len; ++i)
    {
        rgNodeResize(graph, &graph->nodes.ptr[i]);
    }

    for (uint32_t i = 0; i < graph->resources.len; ++i)
    {
        rgResourceResize(graph, &graph->resources.ptr[i]);
    }

    for (uint32_t i = 0; i < graph->passes.len; ++i)
    {
        rgPassResize(graph, &graph->passes.ptr[i]);
    }
}

RgGraph *rgGraphCreate(void)
{
    RgGraph *graph = (RgGraph *)malloc(sizeof(RgGraph));
    memset(graph, 0, sizeof(*graph));

    return graph;
}

RgPassRef rgGraphAddPass(RgGraph *graph, RgPassType type)
{
    RgPassRef ref;
    ref.index = graph->passes.len;

    RgPass pass;
    memset(&pass, 0, sizeof(pass));

    pass.graph = graph;
    pass.type = type;

    arrPush(&graph->passes, pass);

    return ref;
}

RgResourceRef rgGraphAddImage(RgGraph *graph, RgGraphImageInfo *info)
{
    RgResourceRef ref;
    ref.index = graph->resources.len;

    RgResource resource;
    memset(&resource, 0, sizeof(resource));
    resource.type = RG_RESOURCE_IMAGE;

    resource.image_info.scaling_mode = info->scaling_mode;
    resource.image_info.width = info->width;
    resource.image_info.height = info->height;
    resource.image_info.depth = info->depth;
    resource.image_info.sample_count = info->sample_count;
    resource.image_info.mip_count = info->mip_count;
    resource.image_info.layer_count = info->layer_count;
    resource.image_info.aspect = info->aspect;
    resource.image_info.format = info->format;

    arrPush(&graph->resources, resource);

    return ref;
}

RgResourceRef rgGraphAddBuffer(RgGraph *graph, RgBufferInfo *info)
{
    RgResourceRef ref;
    ref.index = graph->resources.len;

    RgResource resource;
    memset(&resource, 0, sizeof(resource));
    resource.type = RG_RESOURCE_IMAGE;

    resource.buffer_info = *info;

    arrPush(&graph->resources, resource);

    return ref;
}

RgResourceRef rgGraphAddExternalImage(RgGraph *graph, RgImage *image)
{
    RgResourceRef ref;
    ref.index = graph->resources.len;

    RgResource resource;
    memset(&resource, 0, sizeof(resource));
    resource.type = RG_RESOURCE_EXTERNAL_IMAGE;

    for (uint32_t i = 0; i < RG_FRAMES_IN_FLIGHT; ++i)
    {
        resource.frames[i].image = image;
    }

    resource.image_info.scaling_mode = RG_GRAPH_IMAGE_SCALING_MODE_ABSOLUTE;
    resource.image_info.width = (float)image->info.width;
    resource.image_info.height = (float)image->info.height;
    resource.image_info.depth = image->info.depth;
    resource.image_info.sample_count = image->info.sample_count;
    resource.image_info.mip_count = image->info.mip_count;
    resource.image_info.layer_count = image->info.layer_count;
    resource.image_info.aspect = image->info.aspect;
    resource.image_info.format = image->info.format;

    arrPush(&graph->resources, resource);

    return ref;
}

RgResourceRef rgGraphAddExternalBuffer(RgGraph *graph, RgBuffer *buffer)
{
    RgResourceRef ref;
    ref.index = graph->resources.len;

    RgResource resource;
    memset(&resource, 0, sizeof(resource));
    resource.type = RG_RESOURCE_EXTERNAL_BUFFER;

    for (uint32_t i = 0; i < RG_FRAMES_IN_FLIGHT; ++i)
    {
        resource.frames[i].buffer = buffer;
    }

    arrPush(&graph->resources, resource);

    return ref;
}

static void ensureResourceUsage(RgResource *res, RgResourceUsage usage)
{
    switch (usage)
    {
    case RG_RESOURCE_USAGE_UNDEFINED: break;

    case RG_RESOURCE_USAGE_COLOR_ATTACHMENT:
    {
        if (res->type == RG_RESOURCE_IMAGE)
        {
            res->image_info.usage |=
                RG_IMAGE_USAGE_COLOR_ATTACHMENT | RG_IMAGE_USAGE_SAMPLED;
            res->image_info.aspect |= RG_IMAGE_ASPECT_COLOR;
        }
        break;
    }

    case RG_RESOURCE_USAGE_SAMPLED:
    {
        if (res->type == RG_RESOURCE_IMAGE)
        {
            res->image_info.usage |= RG_IMAGE_USAGE_SAMPLED;
        }
        break;
    }

    case RG_RESOURCE_USAGE_DEPTH_STENCIL_ATTACHMENT:
    {
        if (res->type == RG_RESOURCE_IMAGE)
        {
            res->image_info.usage |= RG_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT;
            res->image_info.aspect |= RG_IMAGE_ASPECT_DEPTH;
        }
        break;
    }

    case RG_RESOURCE_USAGE_TRANSFER_SRC:
    {
        if (res->type == RG_RESOURCE_IMAGE)
        {
            res->image_info.usage |= RG_IMAGE_USAGE_TRANSFER_SRC;
        }
        if (res->type == RG_RESOURCE_BUFFER)
        {
            res->buffer_info.usage |= RG_BUFFER_USAGE_TRANSFER_SRC;
        }
        break;
    }

    case RG_RESOURCE_USAGE_TRANSFER_DST:
    {
        if (res->type == RG_RESOURCE_IMAGE)
        {
            res->image_info.usage |= RG_IMAGE_USAGE_TRANSFER_DST;
        }
        if (res->type == RG_RESOURCE_BUFFER)
        {
            res->buffer_info.usage |= RG_BUFFER_USAGE_TRANSFER_DST;
        }
        break;
    }
    }
}

void rgGraphPassUseResource(
    RgGraph *graph,
    RgPassRef pass_ref,
    RgResourceRef resource_ref,
    RgResourceUsage pre_usage,
    RgResourceUsage post_usage)
{
    RgPass *pass = &graph->passes.ptr[pass_ref.index];
    RgResource *res = &graph->resources.ptr[resource_ref.index];

    ensureResourceUsage(res, post_usage);

    RgPassResource pass_res;
    pass_res.index = resource_ref.index;
    pass_res.pre_usage = pre_usage;
    pass_res.post_usage = post_usage;

    arrPush(&pass->used_resources, pass_res);
}

void rgGraphBuild(
    RgGraph *graph,
    RgDevice *device,
    RgCmdPool *cmd_pool,
    RgGraphInfo *info)
{
    graph->device = device;
    graph->user_data = info->user_data;
    graph->cmd_pool = cmd_pool;

    graph->num_frames = 1;
    if (info->window)
    {
        assert(info->width > 0 && info->height > 0);
        graph->has_swapchain = true;
        rgSwapchainInit(graph->device, &graph->swapchain, info->window, info->preferred_swapchain_format);
        graph->num_frames = RG_FRAMES_IN_FLIGHT;
    }

    assert(graph->passes.len > 0);
    assert(graph->nodes.len == 0);

    RgNode node;
    memset(&node, 0, sizeof(node));
    arrPush(&graph->nodes, node);

    for (uint32_t i = 0; i < graph->nodes.len; ++i)
    {
        // TODO: split passes across multiple nodes when needed (e.g. compute)
        uint32_t *pass_indices = malloc(sizeof(uint32_t) * graph->passes.len);
        uint32_t num_passes = graph->passes.len;

        for (uint32_t i = 0; i < graph->passes.len; ++i)
        {
            pass_indices[i] = i;
        }

        rgNodeInit(graph, &graph->nodes.ptr[i], pass_indices, num_passes);
    }

    for (uint32_t i = 0; i < graph->nodes.len; ++i)
    {
        RgNode *node = &graph->nodes.ptr[i];
        for (uint32_t j = 0; j < node->num_pass_indices; ++j)
        {
            RgPass *pass = &graph->passes.ptr[node->pass_indices[j]];
            pass->node_index = i;

            if (i == (graph->nodes.len - 1) &&
                node->pass_indices[j] == (graph->passes.len - 1) &&
                graph->has_swapchain)
            {
                pass->is_backbuffer = true;
            }
        }
    }

    for (uint32_t i = 0; i < graph->passes.len; ++i)
    {
        rgPassBuild(graph, &graph->passes.ptr[i]);
    }

    rgGraphResize(graph, info->width, info->height);

    graph->built = true;
}

void rgGraphDestroy(RgGraph *graph)
{
    VK_CHECK(vkDeviceWaitIdle(graph->device->device));

    for (uint32_t i = 0; i < graph->num_frames; ++i)
    {
        vkDestroySemaphore(
            graph->device->device, graph->image_available_semaphores[i], NULL);
    }

    for (uint32_t i = 0; i < graph->nodes.len; ++i)
    {
        rgNodeDestroy(graph, &graph->nodes.ptr[i]);
    }
    arrFree(&graph->nodes);

    for (uint32_t i = 0; i < graph->passes.len; ++i)
    {
        rgPassDestroy(graph, &graph->passes.ptr[i]);
    }

    for (uint32_t f = 0; f < graph->num_frames; ++f)
    {
        for (uint32_t i = 0; i < graph->resources.len; ++i)
        {
            switch (graph->resources.ptr[i].type)
            {
            case RG_RESOURCE_IMAGE:
                if (graph->resources.ptr[i].frames[f].image)
                {
                    rgImageDestroy(graph->device, graph->resources.ptr[i].frames[f].image);
                    graph->resources.ptr[i].frames[f].image = NULL;
                }
                break;
            case RG_RESOURCE_BUFFER:
                if (graph->resources.ptr[i].frames[f].buffer)
                {
                    rgBufferDestroy(graph->device, graph->resources.ptr[i].frames[f].buffer);
                    graph->resources.ptr[i].frames[f].buffer = NULL;
                }
                break;

            case RG_RESOURCE_EXTERNAL_BUFFER:
            case RG_RESOURCE_EXTERNAL_IMAGE: break;
            }
        }
    }

    if (graph->has_swapchain)
    {
        rgSwapchainDestroy(graph->device, &graph->swapchain);
    }

    arrFree(&graph->buffer_barriers);
    arrFree(&graph->image_barriers);
    arrFree(&graph->passes);
    arrFree(&graph->resources);

    free(graph);
}

RgResult rgGraphBeginFrame(RgGraph *graph, uint32_t width, uint32_t height)
{
    if (!graph->has_swapchain) return RG_SUCCESS;

    RgNode *last_node = &graph->nodes.ptr[graph->nodes.len-1];

    VK_CHECK(vkWaitForFences(
        graph->device->device,
        1,
        &last_node->frames[graph->current_frame].fence,
        VK_TRUE,
        UINT64_MAX));

    VkResult res = vkAcquireNextImageKHR(
        graph->device->device,
        graph->swapchain.swapchain,
        UINT64_MAX,
        graph->image_available_semaphores[graph->current_frame],
        VK_NULL_HANDLE,
        &graph->swapchain.current_image_index);
    if ((res == VK_ERROR_OUT_OF_DATE_KHR) || (res == VK_SUBOPTIMAL_KHR))
    {
        rgGraphResize(graph, width, height);
        return RG_RESIZE_NEEDED;
    }
    else
    {
        VK_CHECK(res);
    }

    return RG_SUCCESS;
}

void rgGraphEndFrame(RgGraph *graph, uint32_t width, uint32_t height)
{
    if (!graph->has_swapchain) return;

    RgNode *last_node = &graph->nodes.ptr[graph->nodes.len-1];

    // Last node has to present the swapchain image
    VkPresentInfoKHR present_info;
    memset(&present_info, 0, sizeof(present_info));
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores =
        &last_node->frames[graph->current_frame].execution_finished_semaphore;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &graph->swapchain.swapchain;
    present_info.pImageIndices = &graph->swapchain.current_image_index;

    assert(graph->swapchain.swapchain != VK_NULL_HANDLE);

    // Increment the frame index
    graph->current_frame = (graph->current_frame + 1) % graph->num_frames;

    VkResult res = vkQueuePresentKHR(graph->swapchain.present_queue, &present_info);
    if (!((res == VK_SUCCESS) || (res == VK_SUBOPTIMAL_KHR)))
    {
        if (res == VK_ERROR_OUT_OF_DATE_KHR)
        {
            rgGraphResize(graph, width, height);
        }
        else
        {
            VK_CHECK(res);
        }
    }
}

RgCmdBuffer *rgGraphBeginPass(RgGraph *graph, RgPassRef pass_ref)
{
    RgPass *pass = &graph->passes.ptr[pass_ref.index];
    RgNode *node = &graph->nodes.ptr[pass->node_index];

    uint32_t current_frame = graph->current_frame;
    RgCmdBuffer *cmd_buffer = node->frames[current_frame].cmd_buffer;

    uint32_t first_pass_index = node->pass_indices[0];

    // If the pass is the first in the node, we need to begin
    // the command buffer.
    if (pass_ref.index == first_pass_index)
    {
        // Begin command buffer

        VK_CHECK(
            vkResetFences(graph->device->device, 1, &node->frames[current_frame].fence));

        VkCommandBufferBeginInfo begin_info;
        memset(&begin_info, 0, sizeof(begin_info));
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmd_buffer->cmd_buffer, &begin_info));
    }

    //
    // Apply some barriers
    //

    // TODO: these stages are not optimal
    // We need to base these on the current pass' stage and the next pass' stage
    VkPipelineStageFlags src_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkPipelineStageFlags dst_stage_mask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    graph->buffer_barriers.len = 0;
    graph->image_barriers.len = 0;

    for (uint32_t i = 0; i < pass->used_resources.len; ++i)
    {
        RgPassResource pass_res = pass->used_resources.ptr[i];
        RgResource *resource = &graph->resources.ptr[pass_res.index];

        switch (resource->type)
        {
        case RG_RESOURCE_IMAGE:
        case RG_RESOURCE_EXTERNAL_IMAGE:
        {
            RgImage *image = resource->frames[graph->current_frame].image;
            assert(image);

            VkImageMemoryBarrier barrier;
            memset(&barrier, 0, sizeof(barrier));
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image->image;
            barrier.subresourceRange.aspectMask = image->aspect;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = image->info.mip_count;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = image->info.layer_count;

            rgResourceUsageToVk(pass_res.pre_usage,
                                &barrier.srcAccessMask,
                                &barrier.oldLayout);
            rgResourceUsageToVk(pass_res.post_usage,
                                &barrier.dstAccessMask,
                                &barrier.newLayout);

            switch (pass_res.post_usage)
            {
            case RG_RESOURCE_USAGE_TRANSFER_SRC:
            case RG_RESOURCE_USAGE_TRANSFER_DST:
            {
                arrPush(&graph->image_barriers, barrier);
                break;
            }

            default: break;
            }
            break;
        }

        case RG_RESOURCE_BUFFER:
        case RG_RESOURCE_EXTERNAL_BUFFER:
        {
            RgBuffer *buffer = resource->frames[graph->current_frame].buffer;
            assert(buffer);

            VkBufferMemoryBarrier barrier;
            memset(&barrier, 0, sizeof(barrier));
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = buffer->buffer;
            barrier.offset = 0;
            barrier.size = VK_WHOLE_SIZE;

            rgResourceUsageToVk(pass_res.pre_usage,
                                &barrier.srcAccessMask,
                                NULL);
            rgResourceUsageToVk(pass_res.post_usage,
                                &barrier.dstAccessMask,
                                NULL);

            switch (pass_res.post_usage)
            {
            case RG_RESOURCE_USAGE_TRANSFER_SRC:
            case RG_RESOURCE_USAGE_TRANSFER_DST:
            {
                arrPush(&graph->buffer_barriers, barrier);
                break;
            }

            default: break;
            }
            break;
        }
        }
    }

    if (graph->buffer_barriers.len > 0 || graph->image_barriers.len > 0)
    {
        vkCmdPipelineBarrier(
            cmd_buffer->cmd_buffer,
            src_stage_mask,
            dst_stage_mask,
            0,
            0,
            NULL,
            (uint32_t)graph->buffer_barriers.len,
            graph->buffer_barriers.ptr,
            (uint32_t)graph->image_barriers.len,
            graph->image_barriers.ptr);
    }

    //
    // Begin render pass (if applicable)
    //

    if (pass->type == RG_PASS_TYPE_GRAPHICS)
    {
        if (pass->is_backbuffer)
        {
            assert(graph->has_swapchain);
            assert(pass->num_framebuffers == graph->swapchain.num_images);

            pass->current_framebuffer = pass->frames[graph->current_frame]
                .framebuffers[graph->swapchain.current_image_index];
        }
        else
        {
            assert(pass->num_framebuffers == 1);

            pass->current_framebuffer =
                pass->frames[graph->current_frame].framebuffers[0];
        }

        memset(pass->clear_values, 0, sizeof(VkClearValue) * pass->num_attachments);

        if (pass->has_depth_attachment)
        {
            pass->clear_values[pass->depth_attachment_index].depthStencil.depth =
                1.0f;
            pass->clear_values[pass->depth_attachment_index].depthStencil.stencil = 0;
        }

        VkRenderPassBeginInfo render_pass_info;
        memset(&render_pass_info, 0, sizeof(render_pass_info));
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        render_pass_info.renderPass = pass->renderpass;
        render_pass_info.framebuffer = pass->current_framebuffer;
        render_pass_info.renderArea.offset = (VkOffset2D){0, 0};
        render_pass_info.renderArea.extent = pass->extent;
        render_pass_info.clearValueCount = pass->num_attachments;
        render_pass_info.pClearValues = pass->clear_values;
        assert(pass->extent.width > 0);
        assert(pass->extent.height > 0);

        cmd_buffer->current_pass = pass;
        vkCmdBeginRenderPass(
            cmd_buffer->cmd_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport;
        memset(&viewport, 0, sizeof(viewport));
        viewport.x = 0;
        viewport.y = 0;
        viewport.width = (float)pass->extent.width;
        viewport.height = (float)pass->extent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd_buffer->cmd_buffer, 0, 1, &viewport);

        VkRect2D scissor;
        memset(&scissor, 0, sizeof(scissor));
        scissor.offset.x = 0;
        scissor.offset.y = 0;
        scissor.extent = pass->extent;

        vkCmdSetScissor(cmd_buffer->cmd_buffer, 0, 1, &scissor);
    }

    return cmd_buffer;
}

void rgGraphEndPass(RgGraph *graph, RgPassRef pass_ref)
{
    RgPass *pass = &graph->passes.ptr[pass_ref.index];
    RgNode *node = &graph->nodes.ptr[pass->node_index];

    uint32_t current_frame = graph->current_frame;
    RgCmdBuffer *cmd_buffer = node->frames[current_frame].cmd_buffer;

    if (pass->type == RG_PASS_TYPE_GRAPHICS)
    {
        vkCmdEndRenderPass(cmd_buffer->cmd_buffer);
    }
    cmd_buffer->current_pass = NULL;

    uint32_t last_pass_index = node->pass_indices[node->num_pass_indices-1];
    
    // If the pass is the last in the node, we need to
    // end the command buffer and submit
    if (pass_ref.index == last_pass_index)
    {
        rgBufferPoolReset(&cmd_buffer->ubo_pool);
        rgBufferPoolReset(&cmd_buffer->vbo_pool);
        rgBufferPoolReset(&cmd_buffer->ibo_pool);

        VK_CHECK(vkEndCommandBuffer(cmd_buffer->cmd_buffer));

        VkSubmitInfo submit;
        memset(&submit, 0, sizeof(submit));
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = node->frames[current_frame].wait_semaphores.len;
        submit.pWaitSemaphores = node->frames[current_frame].wait_semaphores.ptr;
        submit.pWaitDstStageMask = node->frames[current_frame].wait_stages.ptr;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd_buffer->cmd_buffer;
        if (pass->is_backbuffer)
        {
            uint32_t num_signal_semaphores = 1;
            VkSemaphore *signal_semaphores =
                &node->frames[current_frame].execution_finished_semaphore;

            submit.signalSemaphoreCount = num_signal_semaphores;
            submit.pSignalSemaphores = signal_semaphores;
        }

        VK_CHECK(vkQueueSubmit(
            graph->device->graphics_queue,
            1,
            &submit,
            node->frames[current_frame].fence));
    }
}

void rgGraphWaitAll(RgGraph *graph)
{
    for (uint32_t i = 0; i < graph->nodes.len; ++i)
    {
        RgNode *node = &graph->nodes.ptr[i];
        VK_CHECK(vkWaitForFences(
            graph->device->device,
            1,
            &node->frames[graph->current_frame].fence,
            VK_TRUE,
            UINT64_MAX));
    }
}

RgBuffer *rgGraphGetBuffer(RgGraph *graph, RgResourceRef resource_ref)
{
    RgResource *resource = &graph->resources.ptr[resource_ref.index];

    switch (resource->type)
    {
    case RG_RESOURCE_EXTERNAL_BUFFER:
    case RG_RESOURCE_BUFFER: {
        assert(resource->frames[graph->current_frame].buffer);
        return resource->frames[graph->current_frame].buffer;
    }

    default: break;
    }

    assert(0);
    return NULL;
}

RgImage *rgGraphGetImage(RgGraph *graph, RgResourceRef resource_ref)
{
    RgResource *resource = &graph->resources.ptr[resource_ref.index];

    switch (resource->type)
    {
    case RG_RESOURCE_EXTERNAL_IMAGE:
    case RG_RESOURCE_IMAGE: {
        assert(resource->frames[graph->current_frame].image);
        return resource->frames[graph->current_frame].image;
    }

    default: break;
    }

    assert(0);
    return NULL;
}
// }}}

#endif // RENDERGRAPH_FEATURE_VULKAN
