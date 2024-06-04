########################################################
# Imports

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from PIL import Image
from glfw import *
import math
import glfw.GLFW as GLFW_CONSTANTS
import sys

########################################################
# Constants

PIPELINE_TYPE : dict[str, int] = {
    "STANDARD" : 0,
    "BLIT" : 1,
    "TINT" : 2,
    "CRT" : 3,
    "POST" : 4,
    "TEXT" : 5,
}

UNIFORM_TYPE = {
    "MODEL":0,
    "VIEW":1,
    "LIGHT_POSITION":2,
    "LIGHT_COLOR":3,
    "LIGHT_STRENGTH":4,
    "CAMERA_POSITION":5,
    "TINT":6,
}

ENTITY_TYPE : dict[str, int] = {
    "CUBE" : 0,
    "MEDKIT" : 1,
    "LIGHT" : 2,
}

SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
RETURN_ACTION_CONTINUE = 0
RETURN_ACTION_END = 1
FPS = 60
FOV = 45

def initialize_glfw():
    init()
    window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR, 3)
    window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR, 3)
    window_hint(GLFW_CONSTANTS.GLFW_OPENGL_PROFILE, GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE)
    window_hint(GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
    window_hint(GLFW_CONSTANTS.GLFW_DOUBLEBUFFER, GLFW_CONSTANTS.GLFW_FALSE)
    window = create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "OpenGL", None, None)
    make_context_current(window)
    set_input_mode(window, GLFW_CONSTANTS.GLFW_CURSOR, GLFW_CONSTANTS.GLFW_CURSOR_HIDDEN)
    
    return window

def create_shader(vertex,fragment):
    with open(vertex,'r') as v:
        vertex = v.readlines()
    with open(fragment,'r') as f:
        fragment = f.readlines()
    return compileProgram(compileShader(vertex,GL_VERTEX_SHADER),
                          compileShader(fragment,GL_FRAGMENT_SHADER))

########################################################
# Shader

class Shader:

    __slots__ = ("program", "multi_uniform", "single_uniform")

    def __init__(self,vertex,fragment):
        self.program = create_shader(vertex,fragment)
        self.multi_uniform :dict[int,list[int]] = {}
        self.single_uniform :dict[int,int] = {}

    def set_single_uniform(self,uniform_type,uniform_name):
        self.single_uniform[uniform_type] = glGetUniformLocation(self.program,uniform_name)

    def set_multi_uniform(self,uniform_type,uniform_name):
        if uniform_type not in self.multi_uniform:
            self.multi_uniform[uniform_type] = []
        self.multi_uniform[uniform_type].append(glGetUniformLocation(self.program,uniform_name))

    def fetch_single_uniform(self,uniform_type):
        return self.single_uniform[uniform_type]
    
    def fetch_multi_uniform(self,uniform_type,index):
        return self.multi_uniform[uniform_type][index]

    def use(self):
        glUseProgram(self.program)

    def destroy(self):
        glDeleteProgram(self.program)

########################################################
# Entities

class Entity:
    def __init__(self,position,eulers):
        self.position = np.array(position,dtype=np.float32)
        self.eulers = np.array(eulers,dtype=np.float32)

    def update(self,rate):
        pass

    def getModelTransform(self):
        pass

class Cube(Entity):
    def __init__(self,position,eulers):
        super().__init__(position,eulers)

    def update(self,rate):
        axis = 1
        self.eulers[axis] = (self.eulers[axis] + rate*0.2) % 360

    def getModelTransform(self,cameraPos):
        model_tranform = pyrr.matrix44.create_identity(dtype = np.float32)
        model_tranform = pyrr.matrix44.multiply(
            m1=model_tranform,
            m2=pyrr.matrix44.create_from_eulers(np.radians(self.eulers),dtype=np.float32)
        )
        model_tranform = pyrr.matrix44.multiply(
            m1=model_tranform,
            m2=pyrr.matrix44.create_from_translation(self.position,dtype=np.float32)
        )
        return model_tranform
        
class Light(Entity):
    def __init__(self,position,color,intensity):
        self.position = np.array(position,dtype=np.float32)
        self.color = np.array(color,dtype=np.float32)
        self.strength = intensity

    def getModelTransform(self,cameraPos):
        directionFromPlayer = self.position - cameraPos
        angle1 = np.arctan2(-directionFromPlayer[1],directionFromPlayer[0])
        dist2d = math.sqrt(directionFromPlayer[0]**2 + directionFromPlayer[1]**2)
        angle2 = np.arctan2(directionFromPlayer[2],dist2d)
        model_tranform = pyrr.matrix44.create_identity(dtype = np.float32)
        model_tranform = pyrr.matrix44.multiply(
            m1=model_tranform,
            m2=pyrr.matrix44.create_from_y_rotation(theta = angle2,dtype=np.float32)
        )
        model_tranform = pyrr.matrix44.multiply(
            m1=model_tranform,
            m2=pyrr.matrix44.create_from_z_rotation(theta = angle1, dtype=np.float32)
        )
        model_tranform = pyrr.matrix44.multiply(
            m1=model_tranform,
            m2=pyrr.matrix44.create_from_translation(
                self.position,dtype=np.float32
            )
        )
        return model_tranform

class Medkit(Entity):
    def __init__(self,position,eulers):
        super().__init__(position,eulers)
    
    def update(self,rate):
        pass

    def getModelTransform(self,cameraPos):
        directionFromPlayer = self.position - cameraPos
        angle1 = np.arctan2(-directionFromPlayer[1],directionFromPlayer[0])
        dist2d = math.sqrt(directionFromPlayer[0]**2 + directionFromPlayer[1]**2)
        angle2 = np.arctan2(directionFromPlayer[2],dist2d)
        model_tranform = pyrr.matrix44.create_identity(dtype = np.float32)
        model_tranform = pyrr.matrix44.multiply(
            m1=model_tranform,
            m2=pyrr.matrix44.create_from_y_rotation(theta = angle2,dtype=np.float32)
        )
        model_tranform = pyrr.matrix44.multiply(
            m1=model_tranform,
            m2=pyrr.matrix44.create_from_z_rotation(theta = angle1, dtype=np.float32)
        )
        model_tranform = pyrr.matrix44.multiply(
            m1=model_tranform,
            m2=pyrr.matrix44.create_from_translation(
                self.position,dtype=np.float32
            )
        )
        return model_tranform

########################################################
# Player and Cameras

class Player:
    def __init__(self, position, eulers):
        self.position = np.array(position, dtype=np.float32)
        self.theta = 0
        self.phi = 0
        self.eulers = np.array(eulers, dtype=np.float32)
        self.update_vectors()

    def update_vectors(self):
        self.forward = np.array([
            np.cos(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
            np.sin(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
            np.sin(np.deg2rad(self.phi)),
        ], dtype=np.float32)

        global_up = np.array([0, 0, 1], dtype=np.float32)

        self.right = np.cross(self.forward, global_up)
        self.up = np.cross(self.right, self.forward)

########################################################
# Scene and Graphics Engine

class Scene:

    __slots__ = ("entities", "player")

    def __init__(self):
        self.entities : dict[int, list[Entity]] = {
            ENTITY_TYPE["CUBE"] : [
                Cube(position=(6, 0, 0), eulers=(0, 0, 0)),
            ],
            ENTITY_TYPE["MEDKIT"] : [
                Medkit(
                    position = [
                        3,2,0
                    ],
                    eulers = [0,0,0]
                ),
                Medkit(
                    position = [
                        3,-2,0
                    ],
                    eulers = [0,0,0]
                ),
            ],
            ENTITY_TYPE["LIGHT"] : [
                Light(
                    position = (np.random.uniform(0,9), np.random.uniform(-2.0,2.0), np.random.uniform(0,4)),
                    color = (np.random.uniform(0,0.9), np.random.uniform(0,1), np.random.uniform(0,1)),
                    intensity = np.random.uniform(3,5)
                )
                for _ in range(8)
            ],
        }

        self.player = Player(position=(0, 0, 2), eulers=(0, 0, 0))

    def update(self,rate):
        for entities in self.entities.values():
            for entity in entities:
                entity.update(rate)

    def move_player(self,dPos):
        dPos = np.array(dPos,dtype=np.float32)
        self.player.position += dPos

    def rotate_player(self, dTheta : float, dPhi : float):
        self.player.theta = (self.player.theta + dTheta) % 360
        self.player.phi = np.clip(self.player.phi + dPhi,-89,89)
        self.player.update_vectors()

class GraphicsEngine:

    #__slots__ = ("shaders", "framebuffers", "materials", "meshes", "screen")

    def __init__(self):
        self._init_opengl()
        
        self._create_assets()

        self._set_one_time_uniforms()
        self._set_multi_time_uniforms()

    def _init_opengl(self):
        glClearColor(0.05,0.16,0.18,1)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA)

    def _create_assets(self):
        self.shaders = {
            PIPELINE_TYPE["STANDARD"] : Shader('shaders/default.vert','shaders/default.frag'),
            PIPELINE_TYPE["BLIT"] : Shader('shaders/screen.vert','shaders/screen.frag'),
            PIPELINE_TYPE["TINT"] : Shader('shaders/lightning.vert','shaders/lightning.frag'),
            PIPELINE_TYPE["CRT"] : Shader('shaders/crt.vert','shaders/crt.frag'),
            PIPELINE_TYPE["POST"] : Shader('shaders/screen.vert','shaders/post.frag'),
            PIPELINE_TYPE["TEXT"] : Shader('shaders/text.vert','shaders/text.frag'),
        }
        self.framebuffers = [
            FrameBuffer([ColorAttachment()],DepthStencilAttachment()),
            FrameBuffer([ColorAttachment()],DepthStencilAttachment()),
        ]


        self.materials : dict[int,Material]= {
            ENTITY_TYPE["CUBE"] : AdvancedMaterial("wood","png"),
            ENTITY_TYPE["MEDKIT"] : AdvancedMaterial("medkit","png"),
            ENTITY_TYPE["LIGHT"] : Material("greenlight","png"),
        }
        self.meshes : dict[int, Mesh] = {
            ENTITY_TYPE["CUBE"] : ObjLoader("models/cube.obj"),
            ENTITY_TYPE["MEDKIT"] : BillBoard(0.6,0.5),
            ENTITY_TYPE["LIGHT"] : BillBoard(0.2,0.1),
        }

        self.font = Font()
        self.text_label = TextLine("CHAN 06",self.font, (-0.9, 0.9), (0.04,0.05))
        self.screen = Screen(0,0,2,2)

    def _set_one_time_uniforms(self):
        shader_type = PIPELINE_TYPE["STANDARD"]
        shader = self.shaders[shader_type]
        shader.use()
        glUniform1i(glGetUniformLocation(shader.program,"material.albedo"),0)
        glUniform1i(glGetUniformLocation(shader.program,"material.ao"),1)
        glUniform1i(glGetUniformLocation(shader.program,"material.normal"),2)
        glUniform1i(glGetUniformLocation(shader.program,"material.specular"),3)
        projection_transforms = pyrr.matrix44.create_perspective_projection_matrix(
            fovy=FOV,aspect = SCREEN_WIDTH/SCREEN_HEIGHT,near=0.1,far=10, dtype = np.float32)
        glUniformMatrix4fv(glGetUniformLocation(shader.program,"projections"),1,GL_FALSE,projection_transforms)
        
        shader_type = PIPELINE_TYPE["TINT"]
        shader = self.shaders[shader_type]
        shader.use()
        glUniform1i(glGetUniformLocation(shader.program,"imageTexture"),0)
        glUniformMatrix4fv(glGetUniformLocation(shader.program,"projections"),1,GL_FALSE,projection_transforms)

        shader_type = PIPELINE_TYPE["CRT"]
        shader = self.shaders[shader_type]
        shader.use()
        glUniform1i(glGetUniformLocation(shader.program,"material"),0)

        shader_type = PIPELINE_TYPE["POST"]
        shader = self.shaders[shader_type]
        shader.use()
        glUniform1i(glGetUniformLocation(shader.program,"material"),0)

        shader_type = PIPELINE_TYPE["TEXT"]
        shader = self.shaders[shader_type]
        shader.use()
        glUniform1i(glGetUniformLocation(shader.program,"screenTexture"),0)

        shader_type = PIPELINE_TYPE["BLIT"]
        shader = self.shaders[shader_type]
        shader.use()
        glUniform1i(glGetUniformLocation(shader.program,"screenTexture"),0)
        
    def _set_multi_time_uniforms(self):
        shader_type = PIPELINE_TYPE["STANDARD"]
        shader = self.shaders[shader_type]
        shader.set_single_uniform(UNIFORM_TYPE["MODEL"],"model")
        shader.set_single_uniform(UNIFORM_TYPE["VIEW"],"view")
        shader.set_single_uniform(UNIFORM_TYPE["CAMERA_POSITION"],"cameraPosition")
        for i in range(8):
            shader.set_multi_uniform(UNIFORM_TYPE["LIGHT_POSITION"],f"Lights[{i}].position")
            shader.set_multi_uniform(UNIFORM_TYPE["LIGHT_COLOR"],f"Lights[{i}].color")
            shader.set_multi_uniform(UNIFORM_TYPE["LIGHT_STRENGTH"],f"Lights[{i}].strength")

        shader_type = PIPELINE_TYPE["TINT"]
        shader = self.shaders[shader_type]
        shader.set_single_uniform(UNIFORM_TYPE["MODEL"],"model")
        shader.set_single_uniform(UNIFORM_TYPE["VIEW"],"view")
        shader.set_single_uniform(UNIFORM_TYPE["TINT"],"tint")

        shader_type = PIPELINE_TYPE["TEXT"]
        shader = self.shaders[shader_type]
        shader.set_single_uniform(UNIFORM_TYPE["TINT"],"textColor")

    def quit(self):
        for mesh in self.meshes.values():
            mesh.destroy()
        for material in self.materials.values():
            material.destroy()
        for framebuffer in self.framebuffers:
            framebuffer.destroy()
        for shader in self.shaders.values():
            shader.destroy()
        self.font.destroy()
        self.text_label.destroy()
        self.screen.destroy()

    def update_fps(self,fps):
        self.text_label = TextLine(f"FPS: {fps}",self.font, (-0.9, 0.9), (0.04,0.05))

    def render(self,scene: Scene):
        #DRAWING FROM PLAYER VIEW#######################
        self.render_from_player_view(scene,scene.player)        
        ################################################
        
        #CRT effect#####################################
        self.crt_effect(_to = 1, _from = 0)
        ################################################

        #DRAWING IN THE DEFAULT FRAMEBUFFER#############
        self.render_from_frame_buffer(_to = 1)
        ################################################
        glFlush()

    def crt_effect(self,_to : int, _from : int):
        self.framebuffers[_to].use()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        shader_type = PIPELINE_TYPE["CRT"]
        shader = self.shaders[shader_type]
        shader.use()
        self.framebuffers[_from].color_attachments[0].use(0)
        self.screen.draw()

    def render_from_frame_buffer(self, _from = 0):
        shader_type = PIPELINE_TYPE["BLIT"]
        shader = self.shaders[shader_type]
        shader.use()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        self.framebuffers[_from].color_attachments[0].use(0)
        self.screen.draw()
        
    def render_from_player_view(self, scene: Scene, Camera : Player):
        self.framebuffers[0].use()
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        shader_type = PIPELINE_TYPE["STANDARD"]
        shader = self.shaders[shader_type]
        shader.use()
        view_transformation = pyrr.matrix44.create_look_at(
            eye = Camera.position,
            target = Camera.position + Camera.forward,
            up = Camera.up,
            dtype = np.float32
        )
        glUniformMatrix4fv(shader.fetch_single_uniform(UNIFORM_TYPE["VIEW"]),1,GL_FALSE,view_transformation)
        for i,light in enumerate(scene.entities[ENTITY_TYPE["LIGHT"]]):
            glUniform3fv(shader.fetch_multi_uniform(UNIFORM_TYPE["LIGHT_POSITION"],i),1,light.position)
            glUniform3fv(shader.fetch_multi_uniform(UNIFORM_TYPE["LIGHT_COLOR"],i) ,1,light.color)
            glUniform1f(shader.fetch_multi_uniform(UNIFORM_TYPE["LIGHT_STRENGTH"],i),light.strength)
        glUniform3fv(shader.fetch_single_uniform(UNIFORM_TYPE["CAMERA_POSITION"]), 1,scene.player.position)

        for entity_type in ENTITY_TYPE.values():
            if entity_type == ENTITY_TYPE["LIGHT"]:
                continue
            for entity in scene.entities[entity_type]:
                self.materials[entity_type].use()
                model_tranform = entity.getModelTransform(scene.player.position)
                glUniformMatrix4fv(shader.fetch_single_uniform(UNIFORM_TYPE["MODEL"]),1,GL_FALSE,model_tranform)
                self.meshes[entity_type].draw()

        shader_type = PIPELINE_TYPE["TINT"]
        shader = self.shaders[shader_type]
        shader.use()
        glUniformMatrix4fv(shader.fetch_single_uniform(UNIFORM_TYPE["VIEW"]),1,GL_FALSE,view_transformation)
        for light in scene.entities[ENTITY_TYPE["LIGHT"]]:
            self.materials[ENTITY_TYPE["LIGHT"]].use()
            glUniform3fv(shader.fetch_single_uniform(UNIFORM_TYPE["TINT"]),1,light.color)
            model_tranform = light.getModelTransform(scene.player.position)
            glUniformMatrix4fv(shader.fetch_single_uniform(UNIFORM_TYPE["MODEL"]),1,GL_FALSE,model_tranform)
            self.meshes[ENTITY_TYPE["LIGHT"]].draw()

        shader_type = PIPELINE_TYPE["TEXT"]
        shader = self.shaders[shader_type]
        shader.use()
        glUniform4fv(shader.fetch_single_uniform(UNIFORM_TYPE["TINT"]),1,np.array((1,0,1,1),dtype=np.float32))
        self.font.use()
        glBindVertexArray(self.text_label.vao)
        glDrawArrays(GL_TRIANGLES,0,self.text_label.vertex_count)

########################################################
# App

class App:

    __slots__ = ("running", "window", "renderer", "scene", "last_time", "current_time", "numFrames", "frameTime", "walk_offset_lookup")

    def __init__(self,window):
        self.running = True
        self.window = window
        self.renderer = GraphicsEngine()
        self.scene = Scene()
        self.last_time = get_time()
        self.current_time = 0
        self.numFrames = 0
        self.frameTime = 0

        self.walk_offset_lookup = {
            1:0,
            2:90,
            3:45,
            4:180,
            6:135,
            7:90,
            8:270,
            9:315,
            11:0,
            12:225,
            13:270,
            14:180,
        }

    def run(self):
        while self.running:
            self.calculateFrameTime()
            self.events()
            self.update()
            self.draw()
        self.quit()

    def events(self):
        if window_should_close(self.window):
            self.running = False
        if get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:
            self.running = False
        
        self.handleKeys()
        self.handleMouse()

        poll_events()
        self.scene.update(self.frameTime / 16.7)
      
    def calculateFrameTime(self):
        self.current_time = get_time()
        delta = self.current_time - self.last_time
        if delta >= 1.0:
            framerate = max(1,int(self.numFrames / delta))
            self.renderer.update_fps(framerate)
            #set_window_title(self.window,f"OpenGL - {framerate} FPS")
            self.last_time = self.current_time
            self.numFrames = -1
            self.frameTime = float(1000.0 / max(1,framerate))
        self.numFrames += 1

    def handleKeys(self):
        keys_dict = {   
        }
        combo = 0
        directionModifier = 0

        if get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_W) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 1
        if get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_A) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 2
        if get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_S) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 4
        if get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_D) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 8

        if combo in self.walk_offset_lookup:
            directionModifier = self.walk_offset_lookup[combo]
            dPos = [
                0.1 * self.frameTime / 16.7 * np.cos(np.deg2rad(self.scene.player.theta + directionModifier)),
                0.1 * self.frameTime / 16.7 * np.sin(np.deg2rad(self.scene.player.theta + directionModifier)),
                0
            ]
            self.scene.move_player(dPos)
    
    def handleMouse(self):
        (x,y) = get_cursor_pos(self.window)
        rate = self.frameTime / 16.7
        theta_increment = -(x - SCREEN_WIDTH/2) * rate
        phi_increment = -(y - SCREEN_HEIGHT/2) * rate
        self.scene.rotate_player(theta_increment,phi_increment)
        set_cursor_pos(self.window,SCREEN_WIDTH/2,SCREEN_HEIGHT/2)

    def quit(self):
        self.renderer.quit()
        sys.exit()

    def update(self):
        pass

    def draw(self):
          self.renderer.render(self.scene)
             
########################################################
# Meshes
class Mesh:
    def __init__(self,vertices):
        self.vertices = np.array(vertices,dtype=np.float32)
        self.vertex_count = len(self.vertices)//8
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER,self.vbo)
        glBufferData(GL_ARRAY_BUFFER,self.vertices.nbytes,self.vertices,GL_STATIC_DRAW)
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,32,ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,32,ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,32,ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)

    def arm_for_drawing(self):
        glBindVertexArray(self.vao)
    
    def draw2(self):
        glDrawArrays(GL_TRIANGLES,0,self.vertex_count)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES,0,self.vertex_count)

    def destroy(self):
        glDeleteVertexArrays(1,(self.vao,))
        glDeleteBuffers(1,(self.vbo,))

class AdvancedMesh:
    def __init__(self,vertices):
        self.vertices = np.array(vertices,dtype=np.float32)
        self.vertex_count = len(self.vertices)//14
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER,self.vbo)
        glBufferData(GL_ARRAY_BUFFER,self.vertices.nbytes,self.vertices,GL_STATIC_DRAW)
        glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,56,ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,56,ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2,3,GL_FLOAT,GL_FALSE,56,ctypes.c_void_p(20))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(3,3,GL_FLOAT,GL_FALSE,56,ctypes.c_void_p(32))
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(4,3,GL_FLOAT,GL_FALSE,56,ctypes.c_void_p(44))
        glEnableVertexAttribArray(4)

    def draw(self):
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES,0,self.vertex_count)

    def destroy(self):
        glDeleteVertexArrays(1,(self.vao,))
        glDeleteBuffers(1,(self.vbo,))

class ObjLoader(AdvancedMesh):
    def __init__(self,filename):
        vertices = self.loadMesh(filename)
        super().__init__(vertices)

    def loadMesh(self,filename):
        vertices = []
        v,vt,vn= [],[],[]
        with open(filename,'r') as file:
            line = file.readline()
            while line:
                words = line.split(" ")
                if words[0] == "v":
                    v.append(self.read_vertex_data(words))
                elif words[0] == "vt":
                    vt.append(self.read_vertex_texture_data(words))
                elif words[0] == "vn":
                    vn.append(self.read_vertex_normal_data(words))
                elif words[0] == "f":
                    self.read_face_data(words,v,vt,vn,vertices)
                line = file.readline()
    
        return vertices

    def read_face_data(self,words,v,vt,vn,vertices):
        triangleCount = len(words) - 3
        for i in range(triangleCount):

            tangent, bitangent = self.get_face_orientation(words, 1, 2 + i, 3 + i, v, vt)

            self.make_corner(words[1],v,vt,vn,vertices,tangent,bitangent)
            self.make_corner(words[2+i],v,vt,vn,vertices,tangent,bitangent)
            self.make_corner(words[3+i],v,vt,vn,vertices,tangent,bitangent)

    def get_face_orientation(self,
        words: list[str], a: int, b: int, c: int, 
        v: list[list[float]], vt: list[list[float]]) -> tuple[list[float]]:
        """
            Get the tangent and bitangent describing the given face.
        """

        v_vt_vn = words[a].split("/")
        pos1 = np.array(v[int(v_vt_vn[0]) - 1], dtype = np.float32)
        uv1 = np.array(vt[int(v_vt_vn[1]) - 1], dtype = np.float32)

        v_vt_vn = words[b].split("/")
        pos2 = np.array(v[int(v_vt_vn[0]) - 1], dtype = np.float32)
        uv2 = np.array(vt[int(v_vt_vn[1]) - 1], dtype = np.float32)

        v_vt_vn = words[c].split("/")
        pos3 = np.array(v[int(v_vt_vn[0]) - 1], dtype = np.float32)
        uv3 = np.array(vt[int(v_vt_vn[1]) - 1], dtype = np.float32)

        #direction vectors
        dPos1 = pos2 - pos1
        dPos2 = pos3 - pos1
        dUV1 = uv2 - uv1
        dUV2 = uv3 - uv1

        # calculate
        den = 1 / (dUV1[0] * dUV2[1] - dUV2[0] * dUV1[1])
        tangent = [0,0,0]
        tangent[0] = den * (dUV2[1] * dPos1[0] - dUV1[1] * dPos2[0])
        tangent[1] = den * (dUV2[1] * dPos1[1] - dUV1[1] * dPos2[1])
        tangent[2] = den * (dUV2[1] * dPos1[2] - dUV1[1] * dPos2[2])

        bitangent = [0,0,0]
        bitangent[0] = den * (-dUV2[0] * dPos1[0] + dUV1[0] * dPos2[0])
        bitangent[1] = den * (-dUV2[0] * dPos1[1] + dUV1[0] * dPos2[1])
        bitangent[2] = den * (-dUV2[0] * dPos1[2] + dUV1[0] * dPos2[2])

        return (tangent, bitangent)

    def make_corner(self,cornerDescription,v,vt,vn,vertices,tangent,bitangent):
        v_vt_vn = cornerDescription.split("/")
        
        for element in v[int(v_vt_vn[0]) - 1]:
            vertices.append(element)
        for element in vt[int(v_vt_vn[1]) - 1]:
            vertices.append(element)
        for element in vn[int(v_vt_vn[2]) - 1]:
            vertices.append(element)
        for element in tangent:
            vertices.append(element)
        for element in bitangent:
            vertices.append(element)

    def read_vertex_data(self,words):
        return [float(words[1]),float(words[2]),float(words[3])]
    
    def read_vertex_texture_data(self,words):
        return [float(words[1]),float(words[2])]
    
    def read_vertex_normal_data(self,words):
        return [float(words[1]),float(words[2]),float(words[3])]

class BillBoard(AdvancedMesh):
    def __init__(self, w, h):
        #x,y,z,u,v,nx,ny,nz
        self.vertices = np.array([
            0, -w/2,  h/2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
            0, -w/2, -h/2, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0,
            0,  w/2, -h/2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0,

            0, -w/2,  h/2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
            0,  w/2, -h/2, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0,
            0,  w/2,  h/2, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0

        ],dtype=np.float32)
        super().__init__(self.vertices)

class Screen(Mesh):
    def __init__(self,x,y,w,h):
        self.vertices = (
            x-w/2, y-h/2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            x+w/2, y-h/2, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
            x+w/2, y+h/2, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,

            x-w/2, y-h/2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            x+w/2, y+h/2, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0,
            x-w/2, y+h/2, 0.0, 0.0, 1.0 ,1.0, 0.0, 0.0
        )
        super().__init__(self.vertices)

########################################################
# Materials

class Material:
    def __init__(self,filepath, filetype):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        with Image.open(f"gfx/{filepath}.{filetype}") as image:
            image_width,image_height = image.size
            image = image.convert("RGBA")
            img_data = bytes(image.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)
    
    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.texture)

    def destroy(self):
        glDeleteTextures(1,(self.texture,))

class AdvancedMaterial:
    """
        A basic texture.
    """


    def __init__(self, filepath: str, filetype):
        """
            Initialize and load the texture.

            Parameters:

                filepath: path to the image file.
        """
        #albedo : 0
        self.textures = []
        self.textures.append(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.textures[0])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        with Image.open(f"gfx/{filepath}_albedo.{filetype}") as image:
            image_width,image_height = image.size
            image = image.convert("RGBA")
            img_data = bytes(image.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        #ambient occlusion : 1
        self.textures.append(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.textures[1])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        with Image.open(f"gfx/{filepath}_ao.{filetype}") as image:
            image_width,image_height = image.size
            image = image.convert("RGBA")
            img_data = bytes(image.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        #normal map : 2
        self.textures.append(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.textures[2])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        with Image.open(f"gfx/{filepath}_normal.{filetype}") as image:
            image_width,image_height = image.size
            image = image.convert("RGBA")
            img_data = bytes(image.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        #specular map : 3
        self.textures.append(glGenTextures(1))
        glBindTexture(GL_TEXTURE_2D, self.textures[3])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        with Image.open(f"gfx/{filepath}_glossmap.{filetype}") as image:
            image_width,image_height = image.size
            image = image.convert("RGBA")
            img_data = bytes(image.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self) -> None:
        """
            Arm the texture for drawing.
        """
        for i in range(len(self.textures)):
            glActiveTexture(GL_TEXTURE0 + i)
            glBindTexture(GL_TEXTURE_2D,self.textures[i])

    def destroy(self) -> None:
        """
            Free the texture.
        """

        glDeleteTextures(len(self.textures), self.textures)

########################################################

class Font:
    def __init__(self):
        #some parameters for fine tuning.
        w = 55.55 / 1000.0
        h =  63.88 / 1150.0
        heightOffset = 8.5 / 1150.0
        margin = 0.014

        """
            Letter: (left, top, width, height)
        """
        self.letterTexCoords = {
            'A': (       w, 1.0 - h,                          w - margin, h - margin), 'B': ( 3.0 * w, 1.0 - h,                          w - margin, h - margin),
            'C': ( 5.0 * w, 1.0 - h,                          w - margin, h - margin), 'D': ( 7.0 * w, 1.0 - h,                          w - margin, h - margin),
            'E': ( 9.0 * w, 1.0 - h,                          w - margin, h - margin), 'F': (11.0 * w, 1.0 - h,                          w - margin, h - margin),
            'G': (13.0 * w, 1.0 - h,                          w - margin, h - margin), 'H': (15.0 * w, 1.0 - h,                          w - margin, h - margin),
            'I': (17.0 * w, 1.0 - h,                          w - margin, h - margin), 'J': (       w, 1.0 - 3.0 * h + heightOffset,     w - margin, h - margin),
            'K': ( 3.0 * w, 1.0 - 3.0 * h + heightOffset,     w - margin, h - margin), 'L': ( 5.0 * w, 1.0 - 3.0 * h + heightOffset,     w - margin, h - margin),
            'M': ( 7.0 * w, 1.0 - 3.0 * h + heightOffset,     w - margin, h - margin), 'N': ( 9.0 * w, 1.0 - 3.0 * h + heightOffset,     w - margin, h - margin),
            'O': (11.0 * w, 1.0 - 3.0 * h + heightOffset,     w - margin, h - margin), 'P': (13.0 * w, 1.0 - 3.0 * h + heightOffset,     w - margin, h - margin),
            'Q': (15.0 * w, 1.0 - 3.0 * h + heightOffset,     w - margin, h - margin), 'R': (17.0 * w, 1.0 - 3.0 * h + heightOffset,     w - margin, h - margin),
            'S': (       w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin), 'T': ( 3.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin),
            'U': ( 5.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin), 'V': ( 7.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin),
            'W': ( 9.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin), 'X': (11.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin),
            'Y': (13.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin), 'Z': (15.0 * w, 1.0 - 5.0 * h + 2 * heightOffset, w - margin, h - margin),

            'a': (       w,                     1.0 - 7.0 * h, w - margin, h - margin), 'b': ( 3.0 * w,         1.0 - 7.0 * h, w - margin, h - margin),
            'c': ( 5.0 * w,                     1.0 - 7.0 * h, w - margin, h - margin), 'd': ( 7.0 * w,         1.0 - 7.0 * h, w - margin, h - margin),
            'e': ( 9.0 * w,                     1.0 - 7.0 * h, w - margin, h - margin), 'f': (11.0 * w,         1.0 - 7.0 * h, w - margin, h - margin),
            'g': (13.0 * w,                     1.0 - 7.0 * h, w - margin, h - margin), 'h': (15.0 * w,         1.0 - 7.0 * h, w - margin, h - margin),
            'i': (17.0 * w,                     1.0 - 7.0 * h, w - margin, h - margin), 'j': (       w,      1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
            'k': ( 3.0 * w,      1.0 - 9.0 * h + heightOffset, w - margin, h - margin), 'l': ( 5.0 * w,      1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
            'm': ( 7.0 * w,      1.0 - 9.0 * h + heightOffset, w - margin, h - margin), 'n': ( 9.0 * w,      1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
            'o': (11.0 * w,      1.0 - 9.0 * h + heightOffset, w - margin, h - margin), 'p': (13.0 * w,      1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
            'q': (15.0 * w,      1.0 - 9.0 * h + heightOffset, w - margin, h - margin), 'r': (17.0 * w,      1.0 - 9.0 * h + heightOffset, w - margin, h - margin),
            's': (       w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin, h - margin), 't': ( 3.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin, h - margin),
            'u': ( 5.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin, h - margin), 'v': ( 7.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin, h - margin),
            'w': ( 9.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin, h - margin), 'x': (11.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin, h - margin),
            'y': (13.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin, h - margin), 'z': (15.0 * w, 1.0 - 11.0 * h + 2 * heightOffset, w - margin, h - margin),

            '0': (       w, 1.0 - 13.0 * h, w - margin, h - margin), '1':  ( 3.0 * w,                1.0 - 13.0 * h, w - margin, h - margin),
            '2': ( 5.0 * w, 1.0 - 13.0 * h, w - margin, h - margin), '3':  ( 7.0 * w,                1.0 - 13.0 * h, w - margin, h - margin),
            '4': ( 9.0 * w, 1.0 - 13.0 * h, w - margin, h - margin), '5':  (11.0 * w,                1.0 - 13.0 * h, w - margin, h - margin),
            '6': (13.0 * w, 1.0 - 13.0 * h, w - margin, h - margin), '7':  (15.0 * w,                1.0 - 13.0 * h, w - margin, h - margin),
            '8': (17.0 * w, 1.0 - 13.0 * h, w - margin, h - margin), '9':  (       w, 1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
            
            '.':  ( 3.0 * w,     1.0 - 15.0 * h + heightOffset, w - margin, h - margin), ',': ( 5.0 * w,     1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
            ';':  ( 7.0 * w,     1.0 - 15.0 * h + heightOffset, w - margin, h - margin), ':': ( 9.0 * w,     1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
            '$':  (11.0 * w,     1.0 - 15.0 * h + heightOffset, w - margin, h - margin), '#': (13.0 * w,     1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
            '\'': (15.0 * w,     1.0 - 15.0 * h + heightOffset, w - margin, h - margin), '!': (17.0 * w,     1.0 - 15.0 * h + heightOffset, w - margin, h - margin),
            '"':  (       w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin), '/': ( 3.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin),
            '?':  ( 5.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin), '%': ( 7.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin),
            '&':  ( 9.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin), '(': (11.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin),
            ')':  (13.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin), '@': (15.0 * w, 1.0 - 17.0 * h + 2 * heightOffset, w - margin, h - margin)
        }

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        with Image.open("gfx/Inconsolata.png", mode = "r") as img:
            image_width,image_height = img.size
            img = img.convert("RGBA")
            img_data = bytes(img.tobytes())
            glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,image_width,image_height,0,GL_RGBA,GL_UNSIGNED_BYTE,img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def get_bounding_box(self,letter):
        if letter not in self.letterTexCoords:
            return None
        return self.letterTexCoords[letter]

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D,self.texture)

    def destroy(self):
        glDeleteTextures(1,(self.texture,))

class TextLine:
    def __init__(self,initial_text,font,start_position,letter_size):
        self.start_position = start_position
        self.letter_size = letter_size
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.build_text(initial_text,font)

    def build_text(self, new_text, font : Font):
        self.vertices = []
        self.vertex_count = 0
        margin_adjustment = 0.96
        for i,letter in enumerate(new_text):
            bounding_box  = font.get_bounding_box(letter)
            if bounding_box is None:
                continue

            #top left
            self.vertices.append(
                self.start_position[0] - self.letter_size[0] + ((2 + margin_adjustment) * i * self.letter_size[0])
            )
            self.vertices.append(self.start_position[1] + self.letter_size[1])
            self.vertices.append(bounding_box[0] - bounding_box[2])
            self.vertices.append(bounding_box[1] + bounding_box[3])
            #top right
            self.vertices.append(
                self.start_position[0] + self.letter_size[0] + ((2 + margin_adjustment) * i * self.letter_size[0])
            )
            self.vertices.append(self.start_position[1] + self.letter_size[1])
            self.vertices.append(bounding_box[0] + bounding_box[2])
            self.vertices.append(bounding_box[1] + bounding_box[3])
            #bottom right
            self.vertices.append(
                self.start_position[0] + self.letter_size[0] + ((2 + margin_adjustment) * i * self.letter_size[0])
            )
            self.vertices.append(self.start_position[1] - self.letter_size[1])
            self.vertices.append(bounding_box[0] + bounding_box[2])
            self.vertices.append(bounding_box[1] - bounding_box[3])

            #bottom right
            self.vertices.append(
                self.start_position[0] + self.letter_size[0] + ((2 + margin_adjustment) * i * self.letter_size[0])
            )
            self.vertices.append(self.start_position[1] - self.letter_size[1])
            self.vertices.append(bounding_box[0] + bounding_box[2])
            self.vertices.append(bounding_box[1] - bounding_box[3])
            #bottom left
            self.vertices.append(
                self.start_position[0] - self.letter_size[0] + ((2 + margin_adjustment) * i * self.letter_size[0])
            )
            self.vertices.append(self.start_position[1] - self.letter_size[1])
            self.vertices.append(bounding_box[0] - bounding_box[2])
            self.vertices.append(bounding_box[1] - bounding_box[3])
            #top left
            self.vertices.append(
                self.start_position[0] - self.letter_size[0] + ((2 + margin_adjustment) * i * self.letter_size[0])
            )
            self.vertices.append(self.start_position[1] + self.letter_size[1])
            self.vertices.append(bounding_box[0] - bounding_box[2])
            self.vertices.append(bounding_box[1] + bounding_box[3])

            self.vertex_count += 6

        self.vertices = np.array(self.vertices, dtype=np.float32)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        offset = 0
        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(offset))
        offset += 8
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(offset))
    
    def destroy(self):
        glDeleteVertexArrays(1,(self.vao,))
        glDeleteBuffers(1,(self.vbo,))

########################################################
# Framebuffers

class ColorAttachment:
    """
        A color buffer which can be attached to a framebuffer.
    """
    __slots__ = ("texture",)


    def __init__(self):
        """
            Initialize the color buffer.
        """

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, SCREEN_WIDTH, SCREEN_HEIGHT)
        glBindTexture(GL_TEXTURE_2D, 0)
    
    def use(self, unit: int = 0) -> None:
        """
            Bind the texture to the given unit.
        """

        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(GL_TEXTURE_2D, self.texture)
    
    def destroy(self) -> None:
        """
            Free the memory.
        """

        glDeleteTextures(1, (self.texture,))

class DepthStencilAttachment:
    """
        A simple renderbuffer which can be used as a depth attachment.
    """
    __slots__ = ("render_buffer",)


    def __init__(self):
        """
            Initialize the buffer.
        """

        self.render_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.render_buffer)
        glRenderbufferStorage(
            GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCREEN_WIDTH, SCREEN_HEIGHT
        )
        glBindRenderbuffer(GL_RENDERBUFFER,0)

    def destroy(self) -> None:
        """
            Free any allocated memory.
        """

        glDeleteRenderbuffers(1, (self.render_buffer,))

class FrameBuffer:
    """
        A framebuffer!
    """
    __slots__ = ("fbo", "color_attachments", "depth_attachment")


    def __init__(self, 
        color_attachments: tuple[ColorAttachment], 
        depth_attachment: DepthStencilAttachment):
        """
            Create a framebuffer with the given attachments.
        """

        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        
        self.color_attachments = color_attachments
        for i,color_buffer in enumerate(color_attachments):
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, 
                GL_TEXTURE_2D, color_buffer.texture, 0)
        
        self.depth_attachment = depth_attachment
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, 
            GL_RENDERBUFFER, depth_attachment.render_buffer)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    def use(self) -> None:
        """
            Bind the framebuffer for drawing.
        """

        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
    
    def destroy(self) -> None:
        """
            Destroy the framebuffer and all its attachments.
        """

        glDeleteFramebuffers(1, (self.fbo,))
        
        for color_buffer in self.color_attachments:
            color_buffer.destroy()
        
        self.depth_attachment.destroy()

########################################################
# Main

if __name__ == '__main__':
    window = initialize_glfw()
    App(window).run()