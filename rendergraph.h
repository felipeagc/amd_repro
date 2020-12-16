#ifndef RENDERGRAPH_H
#define RENDERGRAPH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define RENDERGRAPH_FEATURE_VULKAN

typedef struct RgDevice RgDevice;
typedef struct RgPipeline RgPipeline;
typedef struct RgBuffer RgBuffer;
typedef struct RgImage RgImage;
typedef struct RgSampler RgSampler;
typedef struct RgCmdBuffer RgCmdBuffer;
typedef struct RgCmdPool RgCmdPool;
typedef struct RgGraph RgGraph;
typedef uint32_t RgFlags;

typedef enum RgResult
{
    RG_SUCCESS = 0,
    RG_RESIZE_NEEDED = 1,
} RgResult;

typedef enum RgFormat
{
    RG_FORMAT_UNDEFINED = 0,

    RG_FORMAT_R8_UNORM = 1,
    RG_FORMAT_RG8_UNORM = 2,
    RG_FORMAT_RGB8_UNORM = 3,
    RG_FORMAT_RGBA8_UNORM = 4,

    RG_FORMAT_R8_UINT = 5,
    RG_FORMAT_RG8_UINT = 6,
    RG_FORMAT_RGB8_UINT = 7,
    RG_FORMAT_RGBA8_UINT = 8,

    RG_FORMAT_R16_UINT = 9,
    RG_FORMAT_RG16_UINT = 10,
    RG_FORMAT_RGB16_UINT = 11,
    RG_FORMAT_RGBA16_UINT = 12,

    RG_FORMAT_R32_UINT = 13,
    RG_FORMAT_RG32_UINT = 14,
    RG_FORMAT_RGB32_UINT = 15,
    RG_FORMAT_RGBA32_UINT = 16,

    RG_FORMAT_R32_SFLOAT = 17,
    RG_FORMAT_RG32_SFLOAT = 18,
    RG_FORMAT_RGB32_SFLOAT = 19,
    RG_FORMAT_RGBA32_SFLOAT = 20,

    RG_FORMAT_BGRA8_UNORM = 21,
    RG_FORMAT_BGRA8_SRGB = 22,

    RG_FORMAT_R16_SFLOAT = 23,
    RG_FORMAT_RG16_SFLOAT = 24,
    RG_FORMAT_RGBA16_SFLOAT = 25,

    RG_FORMAT_D32_SFLOAT_S8_UINT = 26,
    RG_FORMAT_D32_SFLOAT = 27,
    RG_FORMAT_D24_UNORM_S8_UINT = 28,
    RG_FORMAT_D16_UNORM_S8_UINT = 29,
    RG_FORMAT_D16_UNORM = 30,

    RG_FORMAT_BC7_UNORM = 31,
    RG_FORMAT_BC7_SRGB = 32,
} RgFormat;

typedef enum RgImageUsage
{
	RG_IMAGE_USAGE_SAMPLED                  = 1 << 0,
	RG_IMAGE_USAGE_TRANSFER_DST             = 1 << 1,
	RG_IMAGE_USAGE_TRANSFER_SRC             = 1 << 2,
	RG_IMAGE_USAGE_STORAGE                  = 1 << 3,
	RG_IMAGE_USAGE_COLOR_ATTACHMENT         = 1 << 4,
	RG_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT = 1 << 5,
} RgImageUsage;

typedef enum RgImageAspect
{
	RG_IMAGE_ASPECT_COLOR   = 1 << 0,
	RG_IMAGE_ASPECT_DEPTH   = 1 << 1,
	RG_IMAGE_ASPECT_STENCIL = 1 << 2,
} RgImageAspect;

typedef enum RgFilter
{
	RG_FILTER_LINEAR = 0,
	RG_FILTER_NEAREST = 1,
} RgFilter;

typedef enum RgSamplerAddressMode
{
    RG_SAMPLER_ADDRESS_MODE_REPEAT = 0,
    RG_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT = 1,
    RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE = 2,
    RG_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER = 3,
    RG_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE= 4,
} RgSamplerAddressMode;

typedef enum RgBorderColor
{
    RG_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK = 0,
    RG_BORDER_COLOR_INT_TRANSPARENT_BLACK = 1,
    RG_BORDER_COLOR_FLOAT_OPAQUE_BLACK = 2,
    RG_BORDER_COLOR_INT_OPAQUE_BLACK = 3,
    RG_BORDER_COLOR_FLOAT_OPAQUE_WHITE = 4,
    RG_BORDER_COLOR_INT_OPAQUE_WHITE = 5,
} RgBorderColor;

typedef struct RgImageInfo
{
	uint32_t width;
	uint32_t height;
	uint32_t depth;
	uint32_t sample_count;
	uint32_t mip_count;
	uint32_t layer_count;
	RgFlags usage;
	RgFlags aspect;
	RgFormat format;
} RgImageInfo;

typedef struct RgSamplerInfo
{
    bool anisotropy;
    float max_anisotropy;
    float min_lod;
    float max_lod;
    RgFilter mag_filter;
    RgFilter min_filter;
    RgSamplerAddressMode address_mode;
    RgBorderColor border_color;
} RgSamplerInfo;

typedef enum RgBufferUsage
{
	RG_BUFFER_USAGE_VERTEX       = 1 << 0,
	RG_BUFFER_USAGE_INDEX        = 1 << 1,
	RG_BUFFER_USAGE_UNIFORM      = 1 << 2,
	RG_BUFFER_USAGE_TRANSFER_SRC = 1 << 3,
	RG_BUFFER_USAGE_TRANSFER_DST = 1 << 4,
	RG_BUFFER_USAGE_STORAGE      = 1 << 5,
} RgBufferUsage;

typedef enum RgBufferMemory
{
	RG_BUFFER_MEMORY_HOST = 1,
	RG_BUFFER_MEMORY_DEVICE,
} RgBufferMemory;

typedef struct RgBufferInfo
{
	size_t size;
	RgFlags usage;
	RgBufferMemory memory;
} RgBufferInfo;

typedef enum RgIndexType
{
    RG_INDEX_TYPE_UINT32 = 0,
    RG_INDEX_TYPE_UINT16 = 1,
} RgIndexType;

typedef enum RgPolygonMode
{
    RG_POLYGON_MODE_FILL = 0,
    RG_POLYGON_MODE_LINE = 1,
    RG_POLYGON_MODE_POINT = 2,
} RgPolygonMode;

typedef enum RgPrimitiveTopology
{
    RG_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 0,
    RG_PRIMITIVE_TOPOLOGY_LINE_LIST = 1,
} RgPrimitiveTopology;

typedef enum RgFrontFace
{
    RG_FRONT_FACE_CLOCKWISE = 0,
    RG_FRONT_FACE_COUNTER_CLOCKWISE = 1,
} RgFrontFace;

typedef enum RgCullMode
{
    RG_CULL_MODE_NONE = 0,
    RG_CULL_MODE_BACK = 1,
    RG_CULL_MODE_FRONT = 2,
    RG_CULL_MODE_FRONT_AND_BACK = 3,
} RgCullMode;

typedef enum RgPipelineBindingType
{
    RG_BINDING_UNIFORM_BUFFER = 1,
    RG_BINDING_STORAGE_BUFFER = 2,
    RG_BINDING_IMAGE = 3,
    RG_BINDING_SAMPLER = 4,
    RG_BINDING_IMAGE_SAMPLER = 5,
} RgPipelineBindingType;

typedef struct RgPipelineBinding
{
    uint32_t set;
    uint32_t binding;
    RgPipelineBindingType type;
} RgPipelineBinding;

typedef struct RgVertexAttribute
{
    RgFormat format;
    uint32_t offset;
} RgVertexAttribute;

typedef struct RgPipelineBlendState
{
    bool enable;
} RgPipelineBlendState;

typedef struct RgPipelineDepthStencilState
{
    bool test_enable;
    bool write_enable;
    bool bias_enable;
} RgPipelineDepthStencilState;

typedef struct RgGraphicsPipelineInfo
{
    RgPolygonMode       polygon_mode;
    RgCullMode          cull_mode;
    RgFrontFace         front_face;
    RgPrimitiveTopology topology;

    RgPipelineBlendState        blend;
    RgPipelineDepthStencilState depth_stencil;

    uint32_t vertex_stride;
    uint32_t num_vertex_attributes;
    RgVertexAttribute *vertex_attributes;

    uint32_t num_bindings;
    RgPipelineBinding *bindings;

    const uint8_t *vertex;
    size_t vertex_size;
    const char* vertex_entry;

    const uint8_t *fragment;
    size_t fragment_size;
    const char* fragment_entry;
} RgGraphicsPipelineInfo;

typedef struct RgComputePipelineInfo
{
    uint32_t num_bindings;
    RgPipelineBinding *bindings;

    const uint8_t *code;
    size_t code_size;
    const char* entry;
} RgComputePipelineInfo;

typedef enum RgResourceUsage
{
    RG_RESOURCE_USAGE_UNDEFINED = 0,
    RG_RESOURCE_USAGE_COLOR_ATTACHMENT = 1,
    RG_RESOURCE_USAGE_DEPTH_STENCIL_ATTACHMENT = 2,
    RG_RESOURCE_USAGE_SAMPLED = 3,
    RG_RESOURCE_USAGE_TRANSFER_SRC = 4,
    RG_RESOURCE_USAGE_TRANSFER_DST = 5,
} RgResourceUsage;

typedef enum RgPassType
{
    RG_PASS_TYPE_GRAPHICS = 0,
    RG_PASS_TYPE_COMPUTE = 1,
    RG_PASS_TYPE_TRANSFER = 2,
} RgPassType;

typedef enum RgGraphImageScalingMode
{
    RG_GRAPH_IMAGE_SCALING_MODE_RELATIVE = 0,
    RG_GRAPH_IMAGE_SCALING_MODE_ABSOLUTE = 1,
} RgGraphImageScalingMode;

typedef struct RgGraphImageInfo
{
    RgGraphImageScalingMode scaling_mode;
	float width;
	float height;
	uint32_t depth;
	uint32_t sample_count;
	uint32_t mip_count;
	uint32_t layer_count;
	RgFlags aspect;
	RgFormat format;
} RgGraphImageInfo;

typedef struct RgOffset3D
{
    int32_t x, y, z;
} RgOffset3D;

typedef struct RgExtent3D
{
    uint32_t width, height, depth;
} RgExtent3D;

typedef struct RgOffset2D
{
    int32_t x, y;
} RgOffset2D;

typedef struct RgExtent2D
{
    uint32_t width, height;
} RgExtent2D;

typedef struct RgRect2D
{
    RgOffset2D offset;
    RgExtent2D extent;
} RgRect2D;

typedef struct RgViewport
{
    float x;
    float y;
    float width;
    float height;
    float min_depth;
    float max_depth;
} RgViewport;

typedef struct RgImageCopy
{
    RgImage *image;
    uint32_t mip_level;
    uint32_t array_layer;
    RgOffset3D offset;
} RgImageCopy;

typedef struct RgBufferCopy
{
    RgBuffer *buffer;
    size_t offset;
    uint32_t row_length;
    uint32_t image_height;
} RgBufferCopy;

typedef struct RgImageRegion
{
    uint32_t base_mip_level;
    uint32_t mip_count;
    uint32_t base_array_layer;
    uint32_t layer_count;
} RgImageRegion;

typedef enum RgObjectType
{
    RG_OBJECT_TYPE_UNKNOWN = 0,
    RG_OBJECT_TYPE_IMAGE = 1,
    RG_OBJECT_TYPE_BUFFER = 2,
} RgObjectType;

typedef struct RgResourceRef
{
    uint32_t index;
} RgResourceRef;

typedef struct RgPassRef
{
    uint32_t index;
} RgPassRef;

typedef enum RgWindowSystem
{
    RG_WINDOW_SYSTEM_NONE,
    RG_WINDOW_SYSTEM_WIN32,
    RG_WINDOW_SYSTEM_X11,
    RG_WINDOW_SYSTEM_WAYLAND,
} RgWindowSystem;

typedef struct RgDeviceInfo
{
    bool enable_validation;
    RgWindowSystem window_system;
} RgDeviceInfo;

typedef struct RgPlatformWindowInfo
{
    struct
    {
        void *window;
        void *display;
    } x11;
    struct
    {
        void *window;
        void *display;
    } wl;
    struct
    {
        void*window;
    } win32;
} RgPlatformWindowInfo;

typedef struct RgGraphInfo
{
    // Dimensions can be zero if not using a swapchain
    uint32_t width;
    uint32_t height;

    // preferred_swapchain_format can be undefined if not using a swapchain
    RgFormat preferred_swapchain_format;

    void *user_data;
    RgPlatformWindowInfo *window;
} RgGraphInfo;

RgDevice *rgDeviceCreate(RgDeviceInfo *info);
void rgDeviceDestroy(RgDevice* device);
void rgDeviceWaitIdle(RgDevice* device);
RgFormat rgDeviceGetSupportedDepthFormat(RgDevice* device, RgFormat wanted_format);

RgCmdPool *rgCmdPoolCreate(RgDevice* device);
void rgCmdPoolDestroy(RgDevice* device, RgCmdPool *cmd_pool);

void rgObjectSetName(RgDevice *device, RgObjectType type, void *object, const char* name);

RgImage *rgImageCreate(RgDevice *device, RgImageInfo *info);
void rgImageDestroy(RgDevice *device, RgImage *image);
void rgImageUpload(RgDevice *device, RgCmdPool *cmd_pool, RgImageCopy *dst, RgExtent3D *extent, size_t size, void *data);
void rgImageBarrier(RgDevice *device, RgCmdPool *cmd_pool, RgImage *image, const RgImageRegion *region, RgResourceUsage from, RgResourceUsage to);
void rgImageGenerateMipMaps(RgDevice *device, RgCmdPool *cmd_pool, RgImage *image);

RgSampler *rgSamplerCreate(RgDevice *device, RgSamplerInfo *info);
void rgSamplerDestroy(RgDevice *device, RgSampler *sampler);

RgBuffer *rgBufferCreate(RgDevice *device, RgBufferInfo *info);
void rgBufferDestroy(RgDevice *device, RgBuffer *buffer);
void *rgBufferMap(RgDevice *device, RgBuffer *buffer);
void rgBufferUnmap(RgDevice *device, RgBuffer *buffer);
void rgBufferUpload(RgDevice *device, RgCmdPool *cmd_pool, RgBuffer *buffer, size_t offset, size_t size, void *data);

RgPipeline *rgGraphicsPipelineCreate(RgDevice *device, RgGraphicsPipelineInfo *info);
RgPipeline *rgComputePipelineCreate(RgDevice *device, RgComputePipelineInfo *info);
void rgPipelineDestroy(RgDevice *device, RgPipeline *pipeline);

RgGraph *rgGraphCreate(void);
RgPassRef rgGraphAddPass(RgGraph *graph, RgPassType pass_type);
RgResourceRef rgGraphAddImage(RgGraph *graph, RgGraphImageInfo *info);
RgResourceRef rgGraphAddBuffer(RgGraph *graph, RgBufferInfo *info);
RgResourceRef rgGraphAddExternalImage(RgGraph *graph, RgImage *image);
RgResourceRef rgGraphAddExternalBuffer(RgGraph *graph, RgBuffer *buffer);
void rgGraphPassUseResource(RgGraph *graph, RgPassRef pass, RgResourceRef resource, RgResourceUsage pre_usage, RgResourceUsage post_usage);
void rgGraphBuild(RgGraph *graph, RgDevice *device, RgCmdPool *cmd_pool, RgGraphInfo *info);
void rgGraphDestroy(RgGraph *graph);
void rgGraphResize(RgGraph *graph, uint32_t width, uint32_t height);
RgResult rgGraphBeginFrame(RgGraph *graph, uint32_t width, uint32_t height);
void rgGraphEndFrame(RgGraph *graph, uint32_t width, uint32_t height);
RgCmdBuffer *rgGraphBeginPass(RgGraph *graph, RgPassRef pass);
void rgGraphEndPass(RgGraph *graph, RgPassRef pass);
void rgGraphWaitAll(RgGraph *graph);
RgBuffer *rgGraphGetBuffer(RgGraph *graph, RgResourceRef resource);
RgImage *rgGraphGetImage(RgGraph *graph, RgResourceRef resource);

void rgCmdSetViewport(RgCmdBuffer *cb, const RgViewport *viewport);
void rgCmdSetScissor(RgCmdBuffer *cb, const RgRect2D *rect);
void rgCmdBindPipeline(RgCmdBuffer *cb, RgPipeline *pipeline);
void rgCmdBindImage(RgCmdBuffer *cb, uint32_t binding, uint32_t set, RgImage *image);
void rgCmdBindSampler(RgCmdBuffer *cb, uint32_t binding, uint32_t set, RgSampler *sampler);
void rgCmdBindImageSampler(RgCmdBuffer *cb, uint32_t binding, uint32_t set, RgImage *image, RgSampler *sampler);
void rgCmdBindVertexBuffer(RgCmdBuffer *cb, RgBuffer *buffer, size_t offset);
void rgCmdBindIndexBuffer(RgCmdBuffer *cb, RgIndexType index_type, RgBuffer *buffer, size_t offset);
void rgCmdBindUniformBuffer(
    RgCmdBuffer *cb,
    uint32_t binding,
    uint32_t set,
    RgBuffer *buffer,
    size_t offset,
    size_t range);
void rgCmdBindStorageBuffer(
    RgCmdBuffer *cb,
    uint32_t binding,
    uint32_t set,
    RgBuffer *buffer,
    size_t offset,
    size_t range);
void rgCmdSetUniform(RgCmdBuffer *cb, uint32_t binding, uint32_t set, size_t size, void *data);
void rgCmdSetVertices(RgCmdBuffer *cb, size_t size, void *data);
void rgCmdSetIndices(RgCmdBuffer *cb, RgIndexType index_type, size_t size, void *data);
void rgCmdDraw(RgCmdBuffer *cb, uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance);
void rgCmdDrawIndexed(RgCmdBuffer *cb, uint32_t index_count, uint32_t instance_count, uint32_t first_index, int32_t vertex_offset, uint32_t first_instance);
void rgCmdDispatch(RgCmdBuffer *cb, uint32_t group_count_x, uint32_t group_count_y, uint32_t group_count_z);
void rgCmdCopyBufferToBuffer(RgCmdBuffer *cb, RgBuffer *src, size_t src_offset, RgBuffer *dst, size_t dst_offset, size_t size);
void rgCmdCopyBufferToImage(RgCmdBuffer *cb, RgBufferCopy *src, RgImageCopy *dst, RgExtent3D extent);
void rgCmdCopyImageToBuffer(RgCmdBuffer *cb, RgImageCopy *src, RgBufferCopy *dst, RgExtent3D extent);
void rgCmdCopyImageToImage(RgCmdBuffer *cb, RgImageCopy *src, RgImageCopy *dst, RgExtent3D extent);

#ifdef __cplusplus
}
#endif

#endif // RENDERGRAPH_H
