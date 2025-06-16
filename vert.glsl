#version 330 core
out vec2 uv;
void main() {
    float x = -1.0 + float((gl_VertexID & 1) << 2);
    float y = -1.0 + float((gl_VertexID & 2) << 1);
    uv = 0.5 + 0.5 * vec2(x, y);
    gl_Position = vec4(x, y, 0.0, 1.0);
}
