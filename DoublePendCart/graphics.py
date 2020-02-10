from vpython import *
import numpy as np
class Sim3D():
    def __init__(self,episode,Database,x_init,len_t,realtime):
        display_instructions(episode+1)
        build_environment()
        self.DB=Database
        self.base_rod_l=20
        self.pend_l=1
        self.end_box_side=0.03
        self.origin=vector(0,0,0)
        self.cart_frame=vector(0,0,0)
        scene.camera.pos=vector(0,1,4)
        self.realtime=realtime
        if not self.realtime:
            self.x_dif_history=np.zeros((len(x_init),len_t))
        # if episode>0:
        #     self.axle.visible = False
        #     self.tip1.visible = False
        #     self.pend1.visible = False
        #     #tip2.visible = False
        #     self.pend2.visible = False
        #     self.tube.visible = False
        #     self.run_title.visible= False
        #     self.samples_title.visible= False

        self.run_title = text(text='#Run {}'.format(episode+1), pos=vec(0,1,-1),align='center',color=color.white, emissive=True)
        self.run_title.height=self.run_title.height*0.4
        self.run_title.length=self.run_title.length*0.4
        if self.DB.db_overflow:
            self.samples_title = text(text='#Samples {}'.format(self.DB.db_dim), pos=vec(0,0.7,-1),align='center',color=color.white, emissive=True)
        else:
            self.samples_title = text(text='#Samples {}'.format(self.DB.db_index), pos=vec(0,0.7,-1),align='center',color=color.white, emissive=True)
        self.samples_title.height=self.samples_title.height*0.2
        self.samples_title.length=self.samples_title.length*0.2
        self.base_rod=cylinder(pos=vector(-self.base_rod_l/2,0,0),axis=vector(self.base_rod_l,0,0),radius=0.005,color=vector(0,0,1))
        self.end_box1=box(pos=vector(-self.base_rod_l/2,0,0),length=self.end_box_side,height=2*self.end_box_side,width=2*self.end_box_side)
        self.end_box2=box(pos=vector(self.base_rod_l/2,0,0),length=self.end_box_side,height=2*self.end_box_side,width=2*self.end_box_side)
        self.tube = extrusion(path=[vec(0,0,0), vec(0.16,0,0)], shape=shapes.circle(radius=0.015, thickness=0.25),\
                            pos=vec(0,0,0), axis=vec(0.02,0,0), color=color.red, end_face_color=color.red)
        self.axle=cylinder(pos=vector(0,0,0.01),axis=vector(0,0,0.02),radius=0.005,color=vector(0,0,1))
        self.pend1=box(pos=vector(0,self.pend_l/2-0.01,0.02),length=0.02,height=self.pend_l,width=0.005, color=color.green)
        #pend2=box(pos=vector(0,1.5*pend_l-0.01,0.02),length=0.02,height=pend_l,width=0.005, color=color.red)
        self.pend2=cylinder(pos=vector(0,1.5*self.pend_l-0.01,0.02),axis=vector(0,self.pend_l,0),radius=0.01,color=color.red)
        self.tip1=sphere(pos=vector(0,self.pend_l,0.02), color=color.green, size=.015*vector(3,2,1))
        #tip2=sphere(pos=vector(0,2*pend_l,0.02), color=color.red, size=.015*vector(3,2,1))
        self.pivot1=vector(self.axle.pos.x, self.axle.pos.y,0)
        self.pend1.rotate(angle=-x_init[1], axis=vec(0,0,1),origin=self.pivot1)
        self.tip1.rotate(angle=-x_init[1], axis=vec(0,0,1),origin=self.pivot1)
        self.pend2.pos=self.tip1.pos
        self.pivot2=vector(self.tip1.pos.x, self.tip1.pos.y,0)
        self.pend2.rotate(angle=-x_init[2], axis=vec(0,0,1),origin=self.pivot2)
        scene.camera.axis=self.tube.pos-scene.camera.pos+vector(0,1,0)
    def update_realtime(self,i,x_dif):
        if self.realtime:
            rate(80)
            #x_dif[0]=0
            self.tube.pos.x+=x_dif[0]
            self.tip1.pos.x+=x_dif[0]
            #tip2.pos.x+=x_dif[0]
            self.axle.pos.x+=x_dif[0]
            self.pend1.pos.x+=x_dif[0]
            #pend2.pos.x+=x_dif[0]
            self.pivot1=vector(self.axle.pos.x, self.axle.pos.y,0)
            self.pend1.rotate(angle=-x_dif[1], axis=vec(0,0,1),origin=self.pivot1)
            self.tip1.rotate(angle=-x_dif[1], axis=vec(0,0,1),origin=self.pivot1)
            self.pend2.pos=self.tip1.pos
            self.pivot2=vector(self.tip1.pos.x, self.tip1.pos.y,0)
            self.pend2.rotate(angle=-x_dif[2], axis=vec(0,0,1),origin=self.pivot2)
        else:
            self.x_dif_history[:,i]=x_dif
    def update(self,i,x_dif):
        if not self.realtime:
            for ii in arange(i):
                rate(80)
                x_dif=self.x_dif_history[:,ii]
                self.tube.pos.x+=x_dif[0]
                self.tip1.pos.x+=x_dif[0]
                #tip2.pos.x+=x_dif[0]
                self.axle.pos.x+=x_dif[0]
                self.pend1.pos.x+=x_dif[0]
                #pend2.pos.x+=x_dif[0]
                self.pivot1=vector(self.axle.pos.x, self.axle.pos.y,0)
                self.pend1.rotate(angle=-x_dif[1], axis=vec(0,0,1),origin=self.pivot1)
                self.tip1.rotate(angle=-x_dif[1], axis=vec(0,0,1),origin=self.pivot1)
                self.pend2.pos=self.tip1.pos
                self.pivot2=vector(self.tip1.pos.x, self.tip1.pos.y,0)
                self.pend2.rotate(angle=-x_dif[2], axis=vec(0,0,1),origin=self.pivot2)


    def reset(self):
        self.axle.visible = False
        self.tip1.visible = False
        self.pend1.visible = False
        #tip2.visible = False
        self.pend2.visible = False
        self.tube.visible = False
        self.run_title.visible= False
        self.samples_title.visible= False









def display_instructions(episode):
    s ="\n Run #{}\n".format(episode)
    s += "Pendulum on Cart:\n"
    s += "    Rotate the camera by dragging with the right mouse button,\n        or hold down the Ctrl key and drag.\n"
    s += "    To zoom, drag with the left+right mouse buttons,\n         or hold down the Alt/Option key and drag,\n         or use the mouse wheel.\n"
    s += "    Shift-drag to pan left/right and up/down.\n"
    s += "Touch screen: pinch/extend to zoom, swipe or two-finger rotate."
    scene.caption = s

def build_environment():
    scene.width = 800
    scene.height = 600
    scene.range = 1
    scene.title = "Pendulum on Cart"
    scene.background=color.black

    grey = color.gray(0.4)
    Nslabs = 10
    R = 10
    w = 5
    d = 0.5
    h = 5
    photocenter = 0.15*w
    hh=1
    # The floor, central post, and ball atop the post
    #floor = box(pos=vector(0,-hh,0),size=vector(.2,24,24), axis=vector(0,1,0), texture=textures.metal)
    lamp = local_light(pos=vector(0,4,0),color=color.gray(0.3))
