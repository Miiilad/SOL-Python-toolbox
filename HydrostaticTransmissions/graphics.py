from vpython import *
import numpy as np
class Sim3D():
    def __init__(self,episode,Database,x_init,len_t,u_lim,realtime):
        self.pad=0.14
        self.lim_bottom=-1
        self.lim_top=10+self.lim_bottom
        self.lim_x=10
        self.lim_z=10
        self.u_lim=u_lim
        display_instructions(episode)
        build_environment(self.lim_bottom,self.lim_top)
        self.DB=Database
        scene.camera.pos=vector(0,3,2.5)
        self.realtime=realtime
        self.ref=vector(0,2.5,0)
        if not self.realtime:
            self.x_dif_history=np.zeros((len(x_init),len_t))
            self.pqr_history=np.zeros((3,len_t))
            self.u_history=np.zeros((4,len_t))
        # if episode>0:
        #     self.axle.visible = False
        #     self.tip1.visible = False
        #     self.pend1.visible = False
        #     #tip2.visible = False
        #     self.pend2.visible = False
        #     self.tube.visible = False
        #     self.run_title.visible= False
        #     self.samples_title.visible= False

        # self.run_title = text(text='#Run {}'.format(episode+1), pos=vec(0,1,-1),align='center',color=color.white, emissive=True)
        # self.run_title.height=self.run_title.height*0.4
        # self.run_title.length=self.run_title.length*0.4
        # if self.DB.db_overflow:
        #     self.samples_title = text(text='#Samples {}'.format(self.DB.db_dim), pos=vec(0,0.7,-1),align='center',color=color.white, emissive=True)
        # else:
        #     self.samples_title = text(text='#Samples {}'.format(self.DB.db_index), pos=vec(0,0.7,-1),align='center',color=color.white, emissive=True)
        # self.samples_title.height=self.samples_title.height*0.2
        # self.samples_title.length=self.samples_title.length*0.2
        # self.base_rod=cylinder(pos=vector(-self.base_rod_l/2,0,0),axis=vector(self.base_rod_l,0,0),radius=0.005,color=vector(0,0,1))

        quad_init_position=vector(x_init[0],x_init[2],x_init[1])
        #base_red=box(pos=quad_init_position,length=0.3,height=0.01,width=0.04,color=color.black)
        self.base_red=sphere(pos=quad_init_position, color=color.black, length=0.35,height=0.02,width=0.05)
        self.base_red.rotate(angle=np.pi/4, axis=vector(0,1,0),origin=quad_init_position)
        self.ring1=ring(pos=self.ref,axis=vector(0,1,0),radius=0.10, thickness=0.02,opacity=0.5)
        self.ring2=ring(pos=self.ref,axis=vector(1,0,0),radius=0.10, thickness=0.02,opacity=0.5)

        #base_green=box(pos=quad_init_position,length=0.04,height=0.011,width=0.3,color=color.black)
        self.base_green=sphere(pos=quad_init_position, color=color.black, length=0.05,height=0.011,width=0.35)
        self.base_green.rotate(angle=np.pi/4, axis=vector(0,1,0),origin=quad_init_position)
        self.tip_x=cylinder(pos=quad_init_position,axis=vector(0.4,0,0),radius=0.002,color=color.red,opacity=0.5)
        self.tip_y=cylinder(pos=quad_init_position,axis=vector(0,0.4,0),radius=0.002,color=color.blue,opacity=0.5)
        self.tip_z=cylinder(pos=quad_init_position,axis=vector(0,0,0.4),radius=0.002,color=color.green,opacity=0.5)
        prop1_pos=vector(0.1,0.02,0.1)+quad_init_position
        prop3_pos=vector(-0.1,0.02,-0.1)+quad_init_position
        prop2_pos=vector(-0.1,0.02,0.1)+quad_init_position
        prop4_pos=vector(0.1,0.02,-0.1)+quad_init_position
        self.prop1=sphere(pos=prop1_pos, color=color.yellow, size=.4*vector(0.25,0.01,0.04))
        self.prop2=sphere(pos=prop2_pos, color=color.yellow, size=.4*vector(0.25,0.01,0.04))
        self.prop3=sphere(pos=prop3_pos, color=color.yellow, size=.4*vector(0.25,0.01,0.04))
        self.prop4=sphere(pos=prop4_pos, color=color.yellow, size=.4*vector(0.25,0.01,0.04))
        self.prop1_base=cylinder(pos=prop1_pos,axis=vector(0,-0.03,0),radius=0.002,color=color.white)
        self.prop2_base=cylinder(pos=prop2_pos,axis=vector(0,-0.03,0),radius=0.002,color=color.white)
        self.prop3_base=cylinder(pos=prop3_pos,axis=vector(0,-0.03,0),radius=0.002,color=color.white)
        self.prop4_base=cylinder(pos=prop4_pos,axis=vector(0,-0.03,0),radius=0.002,color=color.white)
        self.core = extrusion(pos=vector(0,1,0),path=paths.cross(width=0.08, thickness=0.025),shape=shapes.circle(radius=0.015),texture="https://i.imgur.com/zn5QlC8.jpg")

    def update_realtime(self,i,x_dif,u,pqr,h):
        p,q,r=pqr
        if self.realtime:
            #scene.camera.follow(self.base_red)
            if np.linalg.norm([self.base_red.pos.x,self.base_red.pos.y,self.base_red.pos.z])<10:
                scene.camera.follow(self.base_red)
            scene.camera.axis=self.base_red.pos-scene.camera.pos

            self.ring1.rotate(angle=0.1, axis=vector(1,0,0),origin=self.ref)
            self.ring2.rotate(angle=0.02, axis=vector(0,1,0),origin=self.ref)
            error=self.base_red.pos-self.ref
            if np.linalg.norm([error.x,error.y,error.z])<0.1:
                self.ring2.color=color.yellow
                self.ring1.color=color.yellow
            else:
                self.ring2.color=color.white
                self.ring1.color=color.white
            #scene.camera.axis=self.base_red.pos-vector(0,5,5)

            rate(50)
            self.prop_rot(u)
            self.rotate(p*h,vector(1,0,0))
            self.rotate(r*h,vector(0,1,0))
            self.rotate(q*h,vector(0,0,1))
            #print(sol.y[...,-1])
            #print(rot_to_euler(sol.y[9:,-1].reshape(3,3)))
            # k = keysdown() # a list of keys that are down
            # if 'left' in k: v.x -= dv
            # if 'right' in k: v.x += dv
            # if 'down' in k: v.y -= dv
            # if 'up' in k: v.y += dv
            self.move(vector(x_dif[0],x_dif[2],x_dif[1]))
            #print('dif',x_dif)
        else:
            self.x_dif_history[:,i]=x_dif
            self.pqr_history[:,i]=pqr
            self.u_history[:,i]=u
    def update(self,i,x_dif,h):
        if not self.realtime:
            for ii in arange(i):
                p,q,r=self.pqr_history[:,i]
                x_dif=self.x_dif_history[:,i]
                u=self.u_history[:,i]
                if np.linalg.norm([self.base_red.pos.x,self.base_red.pos.y,self.base_red.pos.z])<10:
                    scene.camera.follow(self.base_red)
                scene.camera.axis=self.base_red.pos-scene.camera.pos

                self.ring1.rotate(angle=0.1, axis=vector(1,0,0),origin=self.ref)
                self.ring2.rotate(angle=0.02, axis=vector(0,1,0),origin=self.ref)
                error=self.base_red.pos-self.ref
                if np.linalg.norm([error.x,error.y,error.z])<0.1:
                    self.ring2.color=color.yellow
                    self.ring1.color=color.yellow
                else:
                    self.ring2.color=color.white
                    self.ring1.color=color.white

                rate(50)
                self.prop_rot(u)
                self.rotate(p*h,vector(1,0,0))
                self.rotate(r*h,vector(0,1,0))
                self.rotate(q*h,vector(0,0,1))
                self.move(vector(x_dif[0],x_dif[2],x_dif[1]))
                if self.collision_check():
                    break


    def reset(self):
        self.base_red.visible = False
        self.base_green.visible =False

        self.tip_x.visible = False
        self.tip_y.visible = False
        self.tip_z.visible = False
        self.prop1.visible =False
        self.prop2.visible =False
        self.prop3.visible =False
        self.prop4.visible =False
        self.prop1_base.visible =False
        self.prop2_base.visible =False
        self.prop3_base.visible =False
        self.prop4_base.visible =False
        self.core.visible  =False
        self.ring1.visible=False
        self.ring2.visible=False
        scene.camera.pos=vector(0,3,2.5)
        scene.camera.axis=self.base_red.pos-scene.camera.pos



    def rotate(self,ang,ax):
        quad_init_position=self.base_red.pos
        self.base_red.rotate(angle=ang, axis=ax,origin=quad_init_position)
        self.base_green.rotate(angle=ang, axis=ax,origin=quad_init_position)
        self.tip_x.rotate(angle=ang, axis=ax,origin=quad_init_position)
        self.tip_y.rotate(angle=ang, axis=ax,origin=quad_init_position)
        self.tip_z.rotate(angle=ang, axis=ax,origin=quad_init_position)
        self.core.rotate(angle=ang, axis=ax,origin=quad_init_position)
        self.prop1.rotate(angle=ang, axis=ax,origin=quad_init_position)
        self.prop2.rotate(angle=ang, axis=ax,origin=quad_init_position)
        self.prop3.rotate(angle=ang, axis=ax,origin=quad_init_position)
        self.prop4.rotate(angle=ang, axis=ax,origin=quad_init_position)
        self.prop1_base.rotate(angle=ang, axis=ax,origin=quad_init_position)
        self.prop2_base.rotate(angle=ang, axis=ax,origin=quad_init_position)
        self.prop3_base.rotate(angle=ang, axis=ax,origin=quad_init_position)
        self.prop4_base.rotate(angle=ang, axis=ax,origin=quad_init_position)



    def move(self,dx):
        new_pos=self.base_red.pos+dx
        self.base_red.pos=new_pos
        self.base_green.pos=new_pos
        self.tip_x.pos=new_pos
        self.tip_y.pos=new_pos
        self.tip_z.pos=new_pos
        self.core.pos=new_pos
        self.prop1.pos+=dx
        self.prop2.pos+=dx
        self.prop3.pos+=dx
        self.prop4.pos+=dx
        self.prop1_base.pos+=dx
        self.prop2_base.pos+=dx
        self.prop3_base.pos+=dx
        self.prop4_base.pos+=dx

    def prop_rot(self,rate):
        scale=0.8/self.u_lim #to built visible rotations
        self.prop1.rotate(angle=rate[0]*scale, axis=self.tip_y.axis,origin=self.prop1_base.pos)
        self.prop3.rotate(angle=rate[2]*scale, axis=self.tip_y.axis,origin=self.prop3_base.pos)
        self.prop2.rotate(angle=-rate[1]*scale, axis=self.tip_y.axis,origin=self.prop2_base.pos)
        self.prop4.rotate(angle=-rate[3]*scale, axis=self.tip_y.axis,origin=self.prop4_base.pos)

    def collision_check(self):
        v=vector(0,0,0)
        collision=False
        if (self.base_red.pos.y<(self.lim_bottom+self.pad))|(self.base_red.pos.y>(self.lim_top-self.pad)):
            collision=True
            # v.y=-0.3*v.y
            # while (self.base_red.pos.y<(self.lim_bottom+self.pad))|(self.base_red.pos.y>(self.lim_top-self.pad)):
            #     self.move(v)

        if (self.base_red.pos.x<(-self.lim_x+2*self.pad))|(self.base_red.pos.x>(self.lim_x-2*self.pad)):
            collision=True
            # v.x=-0.3*v.x
            # while (self.base_red.pos.x<(-self.lim_x+2*self.pad))|(self.base_red.pos.x>(self.lim_x-2*self.pad)):
            #     self.move(v)

        if (self.base_red.pos.z<(-self.lim_z+2*self.pad))|(self.base_red.pos.z>(self.lim_z-2*self.pad)):
            collision=True
            # v.z=-0.3*v.z
            # while (self.base_red.pos.z<(-self.lim_z+2*pad))|(self.base_red.pos.z>(self.lim_z-2*self.pad)):
            #     self.move(v)
        return(collision)










def display_instructions(episode):
    s ="\n Run #{}\n".format(episode+1)
    s += "Quadrotor:\n"
    s += "    Rotate the camera by dragging with the right mouse button,\n        or hold down the Ctrl key and drag.\n"
    s += "    To zoom, drag with the left+right mouse buttons,\n         or hold down the Alt/Option key and drag,\n         or use the mouse wheel.\n"
    s += "    Shift-drag to pan left/right and up/down.\n"
    s += "Touch screen: pinch/extend to zoom, swipe or two-finger rotate."
    scene.caption = s

def build_environment(lim_bottom,lim_top):
    scene.width = 1024
    scene.height = 768
    scene.range = 1
    scene.title = "Pendulum on Cart"
    scene.background=color.black

    grey = color.gray(0.4)
    Nslabs = 4
    R =10
    w = 20
    d = 0.5
    h = 10
    photocenter = 0.15*w
    scene.lights = []


    light= local_light(pos=vector(0,-5,0),color=color.gray(0.8))
    #room=box(pos=vector(0,0,0),size=vector(10,10,10), axis=vector(0,1,0), texture="https://i.imgur.com/R9QgU7k.jpg")
    # The floor, central post, and ball atop the post
    floor = box(pos=vector(0,lim_bottom,0),size=vector(.2,24,24), axis=vector(0,1,0), texture="https://images.pexels.com/photos/129731/pexels-photo-129731.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260")
    top = box(pos=vector(0,lim_top,0),size=vector(.2,24,24), axis=vector(0,1,0),texture="https://images.pexels.com/photos/158678/garage-door-texture-wooden-wall-panels-158678.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260")
    #lamp = local_light(pos=vector(20,0,20),color=color.gray(0.4))
    #lamp1 = local_light(pos=vector(-20,0,-20),color=color.gray(0.4))
    light=distant_light(direction=vector(0,100,0), color=color.gray(0.7))
    light=distant_light(direction=vector(1,-1,1), color=color.gray(0.5))
    light=distant_light(direction=vector(-1,-1,1), color=color.gray(0.5))
    light=distant_light(direction=vector(1,-1,-1), color=color.gray(0.5))
    light=distant_light(direction=vector(-1,-1,-1), color=color.gray(0.5))

    # Set up the gray slabs, including a portal
    for i in range(Nslabs):
        theta = i*2*pi/Nslabs
        c = cos(theta)
        s = sin(theta)
        xc = R*c
        zc = R*s

        slab = box(pos=vector(R*c, h/2.+lim_bottom, R*s), axis=vector(c,0,s),
                   size=vector(d,h,w), color=grey, texture="https://images.pexels.com/photos/259915/pexels-photo-259915.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260")

        # T = textures.stucco
        #
        # box(pos=slab.pos,
        #     size=vec(1.1*d,0.9*4*photocenter,0.9*4*photocenter), axis=vec(c,0,s),texture=T)
