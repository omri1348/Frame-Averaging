from visdom import Visdom

viz = None

def get_visdom_connection(server,port):
	global viz
	if viz is None:
		viz = Visdom(server=server,port=port)
	return viz

class Visdomer():
    def __init__(self, server, expname, timestamp, port, do_vis):
        self.vis = get_visdom_connection(server=server, port=port)
        self.env = expname + '_' + timestamp
        self.do_vis = do_vis

    def clear_visdom_env(self):
        self.viz.close(env=self.env, win=None)

    def plot_plotly(self,fig,env,win=None):
        if self.do_vis:
            return self.vis.plotlyplot(fig,env=env,win=win)
        else:
            return None


    def plot_video(self,frames,env,win=None):
        if self.do_vis:
            return self.vis.video(frames,env=env)
        else:
            return None
    def plot_txt(self,text,opts,env):
        if self.do_vis:
            self.vis.text(text=text, opts=opts,env=env)


