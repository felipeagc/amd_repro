#ifndef RENDERGRAPH_EXT_H

#include "rendergraph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RgExtCompiledShader
{
    const uint8_t *code;
    size_t code_size;
    const char* entry_point;
} RgExtCompiledShader;

RgPipeline *rgExtGraphicsPipelineCreateWithShaders(
        RgDevice *device,
        RgExtCompiledShader *vertex_shader,
        RgExtCompiledShader *fragment_shader,
        RgGraphicsPipelineInfo *info);

RgPipeline *rgExtComputePipelineCreateWithShaders(
        RgDevice *device,
        RgExtCompiledShader *shader);

#ifdef __cplusplus
}
#endif

#endif // RENDERGRAPH_EXT_H
