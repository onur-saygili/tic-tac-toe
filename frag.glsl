#version 330 core
in vec2 uv;
out vec4 color;

uniform isamplerBuffer tiles;
uniform int n;
uniform int depth;
uniform int player;
uniform int scope;
uniform float width;

#define EMPTY 3
#define MAX_DEPTH 8

int pown(int a, int n) {
    int b = 1;
    while (n > 0) {
        if ((n & 1) == 1)
            b *= a;
        n >>= 1;
        a *= a;
    }
    return b;
}

int n2 = n * n;
int size = (pown(n2, depth + 1) - 1) / (n2 - 1);
const float W = 1.0 / 3.0;
const vec4 BGCOLOR = vec4(1.0);

vec4 blend(vec4 src, vec4 dst) {
    return mix(dst, src, src.a);
}

bool in_scope(int i) {
    while (scope <= i) {
        if (scope == i)
            return true;
        if (i == 0)
            break;
        i = (i - 1) / n2;
    }
    return false;
}

bool bcircle(vec2 p) {
    const float k = W - 1.0;
    float d = dot(p, p);
    return !(d > 1.0 || d < k*k);
}

bool bcross(vec2 p) {
    const float s = 0.70710678118;
    p = abs(s * vec2(p.x - p.y, p.x + p.y));
    p = (p.y > p.x) ? p.yx : p.xy;
    p -= vec2(2.0*s - 0.5*W, 0.5 * W);
    return max(p.x, p.y) <= 0.0;
}

bool btile(vec2 p, int t) {
    p /= 0.9;
    switch (t) {
        case 0: return bcross(p);
        case 1: return bcircle(p);
    }
    return false;
}

vec4 get_color(vec2 p, int t, int d, bool allowed) {
    const vec4 c[3] = vec4[3](
        vec4(0.0, 0.0, 1.0, 0.18),
        vec4(1.0, 0.0, 0.0, 0.15),
        vec4(0.0, 0.0, 0.0, 0.10)
    );
    if (t < 2) {
        return btile(p, t) ? vec4(c[t].rgb, 1.0) : c[2];
    } else if (d == depth) {
        return allowed ? c[player] : c[2];
    } else {
        return vec4(0.0);
    }
}

void main() {
    vec2 p = 2.0*uv - 1.0;
    vec4 layers[MAX_DEPTH + 1];
    bool blocked = false;
    int d, idx = 0;
    for (d = 0; d <= depth; d++) {
        bool allowed = false;
        int tile = int(texelFetch(tiles, idx).x);
        if (tile != EMPTY)
            blocked = true;
        if (max(abs(p.x), abs(p.y)) >= width)
            break;
        if (!blocked && d == depth)
            allowed = in_scope(idx);
        p /= width;
        float nf = float(n);
        int start = n2 * idx + 1;
        int i = int(0.5*nf*(1.0 - p.y));
        int j = int(0.5*nf*(1.0 + p.x));
        layers[d] = get_color(p, tile, d, allowed);
        p.x += 1.0 - float(2*j + 1) / nf;
        p.y -= 1.0 - float(2*i + 1) / nf;
        p *= n;
        idx = start + n*i + j;
    }
    color = BGCOLOR;
    for (int i = 0; i < d; i++)
        color = blend(layers[d - i - 1], color);
}
