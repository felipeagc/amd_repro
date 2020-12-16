#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#else
#define GLFW_EXPOSE_NATIVE_X11
#endif

#include "glfw/include/GLFW/glfw3.h"
#include "glfw/include/GLFW/glfw3native.h"
#include "rendergraph.h"
#include "rendergraph_ext.h"

typedef struct State
{
    GLFWwindow *window;
    RgDevice *device;
    RgCmdPool *cmd_pool;
    RgGraph *graph;
} State;

static void resize_callback(GLFWwindow* window, int width, int height)
{
    State *state = glfwGetWindowUserPointer(window);
    assert(width > 0);
    assert(height > 0);
    rgGraphResize(state->graph, (uint32_t)width, (uint32_t)height);
}

static uint8_t *read_file(const char* path, size_t *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *data = malloc(*size);
    fread(data, *size, 1, f);

    fclose(f);

    return data;
}

static RgPipeline *create_pipeline(
        RgDevice *device,
        const char *vert_path,
        const char *frag_path)
{
    size_t vert_size = 0;
    uint8_t *vert_spv = read_file(vert_path, &vert_size);
    if (!vert_spv)
    {
        printf("Failed to load shader file: %s\n", vert_path);
        exit(1);
    }

    size_t frag_size = 0;
    uint8_t *frag_spv = read_file(frag_path, &frag_size);
    if (!frag_spv)
    {
        printf("Failed to load shader file: %s\n", frag_path);
        exit(1);
    }

    RgExtCompiledShader vert_shader = {
        .code = vert_spv,
        .code_size = vert_size,
        .entry_point = "vertex",
    };

    RgExtCompiledShader frag_shader = {
        .code = frag_spv,
        .code_size = frag_size,
        .entry_point = "pixel",
    };

    RgPipeline *pipeline = rgExtGraphicsPipelineCreateWithShaders(
            device,
            &vert_shader,
            &frag_shader,
            &(RgGraphicsPipelineInfo){
                .polygon_mode = RG_POLYGON_MODE_FILL,
                .cull_mode = RG_CULL_MODE_NONE,
                .front_face = RG_FRONT_FACE_CLOCKWISE,
                .topology = RG_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                .blend = {
                    .enable = false,
                },
                .depth_stencil = {
                    .test_enable = false,
                    .write_enable = false,
                    .bias_enable = false,
                },
            });

    free(frag_spv);
    free(vert_spv);

    return pipeline;
}

int main()
{
    State *state = malloc(sizeof(*state));
    memset(state, 0, sizeof(*state));

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    state->window = glfwCreateWindow(800, 600, "Test", NULL, NULL);
    glfwSetWindowUserPointer(state->window, state);
    glfwSetFramebufferSizeCallback(state->window, resize_callback);

    state->device = rgDeviceCreate(&(RgDeviceInfo){
        .enable_validation = false,
#ifdef _WIN32
        .window_system = RG_WINDOW_SYSTEM_WIN32,
#else
        .window_system = RG_WINDOW_SYSTEM_X11,
#endif
    });

    state->cmd_pool = rgCmdPoolCreate(state->device);

    RgSampler *sampler = rgSamplerCreate(state->device, &(RgSamplerInfo){
        .mag_filter = RG_FILTER_LINEAR,
        .min_filter = RG_FILTER_LINEAR,
        .address_mode = RG_SAMPLER_ADDRESS_MODE_REPEAT,
        .border_color = RG_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
    });

    RgPipeline *color_pipeline = create_pipeline(
            state->device,
            "color.vert.spv",
            "color.frag.spv");

    RgPipeline *post_process_pipeline = create_pipeline(
            state->device,
            "post.vert.spv",
            "post.frag.spv");

    int width, height;
    glfwGetFramebufferSize(state->window, &width, &height);

    state->graph = rgGraphCreate();

    RgResourceRef depth_ref = rgGraphAddImage(state->graph, &(RgGraphImageInfo){
        .scaling_mode = RG_GRAPH_IMAGE_SCALING_MODE_RELATIVE,
        .width = 1.0,
        .height = 1.0,
        .depth = 1,
        .sample_count = 1,
        .mip_count = 1,
        .layer_count = 1,
        .aspect = RG_IMAGE_ASPECT_DEPTH | RG_IMAGE_ASPECT_STENCIL,
        .format = rgDeviceGetSupportedDepthFormat(
                state->device, RG_FORMAT_D32_SFLOAT_S8_UINT),
    });

    RgResourceRef color_ref = rgGraphAddImage(state->graph, &(RgGraphImageInfo){
        .scaling_mode = RG_GRAPH_IMAGE_SCALING_MODE_RELATIVE,
        .width = 1.0,
        .height = 1.0,
        .depth = 1,
        .sample_count = 1,
        .mip_count = 1,
        .layer_count = 1,
        .aspect = RG_IMAGE_ASPECT_COLOR,
        .format = RG_FORMAT_RGBA8_UNORM,
    });

    RgPassRef main_pass = rgGraphAddPass(state->graph, RG_PASS_TYPE_GRAPHICS);
    rgGraphPassUseResource(
            state->graph,
            main_pass,
            color_ref,
            RG_RESOURCE_USAGE_UNDEFINED,
            RG_RESOURCE_USAGE_COLOR_ATTACHMENT);
    rgGraphPassUseResource(
            state->graph,
            main_pass,
            depth_ref,
            RG_RESOURCE_USAGE_UNDEFINED,
            RG_RESOURCE_USAGE_DEPTH_STENCIL_ATTACHMENT);

    RgPassRef backbuffer_pass = rgGraphAddPass(state->graph, RG_PASS_TYPE_GRAPHICS);
    rgGraphPassUseResource(
            state->graph,
            backbuffer_pass,
            color_ref,
            RG_RESOURCE_USAGE_COLOR_ATTACHMENT,
            RG_RESOURCE_USAGE_SAMPLED);

    rgGraphBuild(state->graph, state->device, state->cmd_pool, &(RgGraphInfo){
        .width = width,
        .height = height,

        .preferred_swapchain_format = RG_FORMAT_BGRA8_SRGB,
        .window = &(RgPlatformWindowInfo) {
#ifdef _WIN32
            .win32.window = (void*)glfwGetWin32Window(state->window),
#else
            .x11.window = (void*)glfwGetX11Window(state->window),
            .x11.display = (void*)glfwGetX11Display(),
#endif
        },
    });

    while (!glfwWindowShouldClose(state->window))
    {
        glfwPollEvents();

        int width, height;
        glfwGetFramebufferSize(state->window, &width, &height);
        assert(width > 0);
        assert(height > 0);

        if (rgGraphBeginFrame(state->graph, width, height) == RG_RESIZE_NEEDED)
        {
            continue;
        }

        {
            RgCmdBuffer *cb = rgGraphBeginPass(state->graph, main_pass);

            rgCmdBindPipeline(cb, color_pipeline);
            rgCmdDraw(cb, 3, 1, 0, 0);

            rgGraphEndPass(state->graph, main_pass);
        }

        {
            RgCmdBuffer *cb = rgGraphBeginPass(state->graph, backbuffer_pass);

            RgImage *color_image = rgGraphGetImage(state->graph, color_ref);

            rgCmdBindImage(cb, 0, 0, color_image);
            rgCmdBindSampler(cb, 1, 0, sampler);
            rgCmdBindPipeline(cb, post_process_pipeline);
            rgCmdDraw(cb, 3, 1, 0, 0);

            rgGraphEndPass(state->graph, backbuffer_pass);
        }

        rgGraphEndFrame(state->graph, width, height);
    }

    rgGraphDestroy(state->graph);
    rgPipelineDestroy(state->device, post_process_pipeline);
    rgPipelineDestroy(state->device, color_pipeline);
    rgSamplerDestroy(state->device, sampler);
    rgCmdPoolDestroy(state->device, state->cmd_pool);
    rgDeviceDestroy(state->device);
    glfwDestroyWindow(state->window);
    glfwTerminate();
    free(state);

    return 0;
}
