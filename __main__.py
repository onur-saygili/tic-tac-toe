import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
import numpy as np
from os import environ
from enum import IntEnum
import argparse

# Related to the spacing between tiles (very sensitive to change)
WIDTH = 0.91
# Number of samples to use for anti-aliasing
SAMPLES = 9

class State(IntEnum):
    CROSS = 0
    CIRCLE = 1
    DRAW = 2
    EMPTY = 3

class TicTacToe:
    """A recursive game of tic-tac-toe.

    Attributes:
        n: The dimensions of the board (n x n)
        stride: n^2
        depth: Recursion depth
        size: Total number of tiles (1 + n^2 + n^4 + ... + n^(2d))
        tiles: The game state, stored as a complete (n^2)-ary tree
        width: Related to spacing between tiles
        scope: The index of the tile where the next player can make a move in
    """
    def __init__(self, n = 2, depth = 0, width = 1.0):
        self._n = n
        self._stride = n*n
        self._depth = depth
        self._size = (pow(self._stride, depth + 1) - 1) // (self._stride - 1)
        self._tiles = np.full((self._size,), State.EMPTY, dtype=np.int8)
        self._width = width
        self._scope = 0

    def tiles(self):
        return self._tiles

    def parent(self, i):
        """Return the index of the parent tile"""
        if i > 0 and i < self._size:
            return (i - 1) // self._stride
        else:
            return None

    def children(self, i):
        """Return the index of the first child"""
        j = self._stride * i + 1
        if j > 0 and j < self._size:
            return j
        else:
            return None

    def first(self, i):
        """Return the index of the first child of the parent"""
        if i > 0 and i < self._size:
            l = self._stride
            return ((i - 1) // l) * l + 1
        return 0

    def get_scope(self):
        return self._scope

    def in_scope(self, i):
        """Return if tile is a descendant of scope"""
        while self._scope <= i:
            if self._scope == i:
                return True
            i = self.parent(i)
            if i is None:
                break
        return False

    def check(self, i, v):
        """Return what the state of the parent would be if the tile's state was v"""
        if i == 0 or v == State.EMPTY:
            return v

        n = self._n
        first = self.first(i)
        k = i - first
        tiles = self._tiles[first : first + self._stride]

        if v == State.CROSS or v == State.CIRCLE:
            win = True
            c = k % n # column index
            for j in range(0, n):
                win &= (v == tiles[c + j*n])
            if win:
                return v

            win = True
            r = k // n # row index
            for j in range(0, n):
                win &= (v == tiles[r*n + j])
            if win:
                return v

            if r == c:
                win = True
                for j in range(0, n):
                    win &= (v == tiles[(n + 1)*j])
                if win:
                    return v

            if r + c == n - 1:
                win = True
                for j in range(0, n):
                    win &= (v == tiles[(n - 1)*(j + 1)])
                if win:
                    return v

        draw = True
        for j in range(0, self._stride):
            draw &= (tiles[j] != State.EMPTY)
        if draw:
            return State.DRAW

        return State.EMPTY

    def put(self, i, v):
        """Put tile into state v, update its parent, and set scope accordingly"""
        tiles = self._tiles
        tiles[i] = v
        parent = self.parent(i)
        if parent is None:
            return True
        if v == State.EMPTY:
            return self.put(parent, v)
        result = self.check(i, v)
        if result != State.EMPTY and self.put(parent, result):
            return True
        grandparent = self.parent(parent)
        if grandparent is None:
            return False
        if v == State.DRAW:
            if result == v:
                self._scope = grandparent
                return True
            return False
        j = i - self.first(i) + self.children(grandparent)
        self._scope = tiles[j] == State.EMPTY and j or grandparent
        return True

    def index(self, x, y, idx = 0):
        """Get index of tile at position if a move can be made at that tile"""
        if idx < 0 or idx >= self._size:
            return None
        if self._tiles[idx] != State.EMPTY:
            return None
        w = self._width
        if abs(x) >= w or abs(y) >= w:
            return None
        start = self.children(idx)
        if start is None:
            if self.in_scope(idx):
                return idx
            return None
        n = self._n
        x /= w
        y /= w
        i = int(0.5*n*(1 - y))
        j = int(0.5*n*(1 + x))
        x += 1 - (2*j + 1) / n
        y -= 1 - (2*i + 1) / n
        return self.index(n*x, n*y, start + n*i + j)

def compile_shaders(vertex_source, fragment_source):
    vertex = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex, vertex_source)
    glCompileShader(vertex)

    if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(vertex).decode()
        print(f"Vertex shader compilation error:\n{error}")
        return None

    fragment = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment, fragment_source)
    glCompileShader(fragment)

    if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(fragment).decode()
        print(f"Fragment shader compilation error:\n{error}")
        return None

    program = glCreateProgram()
    glAttachShader(program, vertex)
    glAttachShader(program, fragment)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        print(f"Shader program linking error:\n{error}")
        return None

    glDeleteShader(vertex)
    glDeleteShader(fragment)

    return program

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Ultimate tic-tac-toe with custom depth and board size"
    )
    parser.add_argument(
        "-n", "--size",
        type = int,
        default = 3,
        help = "Size of the Tic-Tac-Toe board (default: 3)"
    )
    parser.add_argument(
        "-d", "--depth",
        type = int,
        default = 2,
        help = "Size of the Tic-Tac-Toe board (default: 2)"
    )
    parser.add_argument(
        "-r", "--res",
        type = int,
        default = 800,
        help = "Window resolution (pixels)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    if args.size < 2:
        raise ArgumentError("Board size must be at least 2.")
    if args.depth < 0 or args.depth > 8:
        raise ArgumentError("Depth must be at least 0 and at most 8.")
    if args.res < 64:
        raise ArgumentError("Resolution must be at least 64.")

    # fix for https://github.com/pygame/pygame/issues/3110
    if environ.get("XDG_SESSION_TYPE") == "wayland":
        environ["SDL_VIDEO_X11_FORCE_EGL"] = "1"

    pg.init()

    # Set OpenGL version to 3.3 core profile
    profile = (3, 3, pg.GL_CONTEXT_PROFILE_CORE)
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, profile[0])
    pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, profile[1])
    pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, profile[2])

    # Create window
    pg.display.set_mode((args.res, args.res), DOUBLEBUF | OPENGL)

    with open("vert.glsl", 'r') as file:
        vert = file.read()
    with open("frag.glsl", 'r') as file:
        frag = file.read()
    with open("post.glsl", 'r') as file:
        blit = file.read()
    drawsh = compile_shaders(vert, frag)
    postsh = compile_shaders(vert, blit)
    if drawsh is None or postsh is None:
        pg.quit()
        return

    res = args.res
    game = TicTacToe(args.size, args.depth, WIDTH)
    tiles = game.tiles()

    u_n = glGetUniformLocation(drawsh, "n")
    u_width = glGetUniformLocation(drawsh, "width")
    u_depth = glGetUniformLocation(drawsh, "depth")
    u_player = glGetUniformLocation(drawsh, "player")
    u_scope = glGetUniformLocation(drawsh, "scope")

    glUseProgram(drawsh)
    glUniform1f(u_width, WIDTH)
    glUniform1i(u_n, args.size)
    glUniform1i(u_depth, args.depth)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    buf = glGenBuffers(1)
    glBindBuffer(GL_TEXTURE_BUFFER, buf)
    glBufferData(GL_TEXTURE_BUFFER, tiles.nbytes, tiles, GL_DYNAMIC_DRAW)

    buftex = glGenTextures(1);
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_BUFFER, buftex);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R8I, buf);

    msaatex = glGenTextures(1);
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, msaatex);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, SAMPLES, GL_RGB, res, res, GL_TRUE);

    msaafbo = glGenFramebuffers(1);
    glBindFramebuffer(GL_FRAMEBUFFER, msaafbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, msaatex, 0);
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("Framebuffer not complete")

    tex = glGenTextures(1);
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, res, res, 0, GL_RGB, GL_UNSIGNED_BYTE, None);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    fbo = glGenFramebuffers(1);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError("Framebuffer not complete")

    player = 0
    running = True
    redraw = True
    while running:
        for event in pg.event.get():
            x, y = pg.mouse.get_pos()
            x = 2 * x / res - 1
            y = 1 - 2 * y / res
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.MOUSEBUTTONUP:
                i = game.index(x, y)
                if i is None:
                    continue
                game.put(i, player)
                player = (player + 1) % 2
                redraw = True

        glViewport(0, 0, res, res)

        # Only render when game state is changed
        if redraw:
            glUseProgram(drawsh)
            glUniform1i(u_player, player)
            glUniform1i(u_scope, game.get_scope())
            glBufferSubData(GL_TEXTURE_BUFFER, 0, tiles.nbytes, tiles)

            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, msaafbo)
            glDrawArrays(GL_TRIANGLES, 0, 3)

            glBindFramebuffer(GL_READ_FRAMEBUFFER, msaafbo)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo)
            glBlitFramebuffer(0, 0, res, res, 0, 0, res, res, GL_COLOR_BUFFER_BIT, GL_LINEAR);

        glUseProgram(postsh)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        pg.display.flip()
        pg.time.wait(16)

    # Clean up
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [buf])
    glDeleteTextures(3, [tex, msaatex, buftex])
    glDeleteFramebuffers(2, [fbo, msaafbo])
    glDeleteProgram(drawsh)
    glDeleteProgram(postsh)

    pg.quit()

if __name__ == "__main__":
    main()
