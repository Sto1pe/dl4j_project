import javax.swing.*;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.util.Hashtable;

public class Visualizer extends JFrame {
    public Plotter p;
    public GamePanel g;
    static final int FPSMAX = 61;
    static final int FPSMIN = 1;
    public int fps= 30;
    public long drawDelayms = (1000/fps);
    public Visualizer(){
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }

    public void addPlotter(int maxGraphs, int datasize, int maxY, String xLabel, String yLabel){
        p = new Plotter(maxGraphs, datasize,maxY, xLabel, yLabel);
        p.setLayout(null);
        this.getContentPane().add(p);
        this.pack();
    }

    public void addGamePanel(float[][] state){
        if(g == null){
            g = new GamePanel(state);
            this.getContentPane().add(g);
            this.pack();
        }else{
            g.setState(state);
        }
    }

    public void addFPSSlider(){
        this.setLayout(new FlowLayout());
        JSlider fpsSlider = new JSlider(JSlider.HORIZONTAL, Visualizer.FPSMIN, Visualizer.FPSMAX, this.fps);
        fpsSlider.setMajorTickSpacing(10);
        fpsSlider.setMinorTickSpacing(1);
        Hashtable labels = fpsSlider.createStandardLabels(10);
        labels.remove(new Integer(FPSMAX));
        labels.put(new Integer(FPSMAX), new JLabel("MAX"));
        fpsSlider.setLabelTable(labels);
        fpsSlider.setToolTipText("FPS");
        fpsSlider.setPaintTicks(true);
        fpsSlider.setPaintLabels(true);
        fpsSlider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider source = (JSlider) e.getSource();
                fps = source.getValue();
                drawDelayms = (1000/fps);
                System.out.println("fps is now: " + fps);
                if(fps == Visualizer.FPSMAX){
                    drawDelayms = 0;
                    g.Render = false;
                }else if(!g.Render){
                    g.Render = true;
                }
            }
        });
        this.getContentPane().add(fpsSlider);
        this.pack();
    }

    public void makeVisible(){
        this.setVisible(true);
    }


    public class Plotter extends JPanel{
        int size;
        Graph[] graphs;
        String xlabel, ylabel, title;
        int xdim, ydim, yzero, xzero, xdraw, ydraw, graphcounter = 0;
        double xtic, ytic, xpoint, ypoint;
        double xmax, xmin, ymax, ymin;

        public Plotter(int maxGraphs, int maxDataSize, int maxY, String xLabel, String yLabel)
        {
            size = maxDataSize;
            graphs = new Graph[maxGraphs];
            xdim = 1800;
            ydim = 1000;
            setPreferredSize(new Dimension(xdim,ydim));
            if(maxDataSize/5 <= 0){
                xtic = 1;
                ytic = 1;
            }else{
                xtic = maxDataSize/5;
                ytic = maxY/5;
            }

            xlabel = (xLabel);
            ylabel = (yLabel);
            title = ylabel + " versus " + xlabel;

            xmax = maxDataSize;
            xmin = 0;
            ymax = maxY;
            ymin = 0;

            //Find Zero point on Ys-axis required for drawing the axes

            if ((ymax*ymin) < 0){
                yzero = (int) ((ydim - 50) - (((0-ymin)/(ymax-ymin)) * (ydim-100)));
            }
            else{
                yzero = (int) ((ydim - 50) - ((0/(ymax-ymin)) * (ydim-100)));
            }

//Find zero point on Xs-axis required for drawing the axes

            if ((xmax*xmin) < 0) {
                xzero = (int) (50 + (((0-xmin)/(xmax-xmin)) * (xdim-100)));
            }
            else{
                xzero = (int) (50 + ((0/(xmax-xmin)) * (xdim-100)));
            }
//Now ready to plot the results
            repaint();
        }

        public void setTitle(String Title){
            title = Title;
        }

        public void addGraph(int datasize, boolean Bargraph){
            graphs[graphcounter] = new Graph(this, datasize, Bargraph);
            graphcounter++;
        }

        public void addPoint(int graphIndex, int x, int y){
            graphs[graphIndex].addPoint(x,y);
            repaint();
        }


        public void paint(Graphics g){
            Font f1 = new Font("TimesRoman", Font.PLAIN, 10);
            g.setFont(f1);

//First draw the axes

//Ys-axis

            g.drawLine(xzero, 50, xzero, ydim-50);
            g.drawLine(xzero, 50, (xzero - 5), 55);
            g.drawLine(xzero, 50, (xzero + 5), 55);

//Xs-axis

            g.drawLine(50, yzero, xdim-50, yzero);
            g.drawLine((xdim-50), yzero, (xdim-55), (yzero + 5));
            g.drawLine((xdim-50), yzero, (xdim-55), (yzero - 5));

//Initialise the labelling taking into account the xtic and ytic values

            //Xs-axis labels

            if (xmin <= 0){
                xpoint = xmin - (xmin%xtic);
            }else{
                xpoint = xmin - (xmin%xtic) + xtic;
            }

            do{
                xdraw = (int) (50 + (((xpoint-xmin)/(xmax-xmin))*(xdim-100)));
                g.drawString(xpoint + "", xdraw, (yzero+10));
                xpoint = xpoint + xtic;
            }while (xpoint <= xmax);

            if (ymin <= 0){
                ypoint = ymin - (ymin%ytic);
            }else{
                ypoint = ymin - (ymin%ytic) + ytic;
            }

            do{
                ydraw = (int) ((ydim - 50) - (((ypoint - ymin)/(ymax-ymin))*(ydim-100)));
                g.drawString(ypoint + "", (xzero - 20), ydraw);
                ypoint = ypoint + ytic;
            }while (ypoint <= ymax);
//Titles and labels
            Font f2 = new Font("TimesRoman", Font.BOLD, 14);
            g.setFont(f2);
            g.drawString(xlabel, (xdim - 100), (yzero + 25));
            g.drawString(ylabel, (xzero - 25), 40);
            g.drawString(title, (xdim/2 - 75), 20);

            for(Graph graph : graphs){
                if(graph != null){
                    graph.paint(g);
                }
            }

        }
    }

    public class Graph{
        int[] Xs;
        int[] Ys;
        int[] xx;
        int[] yy;
        int counter = 0;
        int yymin;
        Plotter Parent;
        String name;
        Color color;
        boolean BarGraph;

        public Graph(Plotter parent, int maxDataSize, boolean barGraph){
            Xs = new int[maxDataSize];
            Ys = new int[maxDataSize];
            BarGraph = barGraph;
            Parent = parent;
            yymin = (int) ((Parent.ydim - 50) - (((0 - Parent.ymin) / (Parent.ymax - Parent.ymin)) * (Parent.ydim - 100)));
            for(int i = 0; i < maxDataSize; i++){
                Xs[i] = 0;
                Ys[i] = 0;
            }
            xx = new int[maxDataSize];
            yy = new int[maxDataSize];
        }

        public void addPoint(int x, int y){
            Xs[counter] = x;
            Ys[counter] = y;
            if (x > Parent.xmax) {
                Parent.xmax = x;
            }
            if (x < Parent.xmin) {
                Parent.xmin = x;
            }
            if (y > Parent.ymax) {
                Parent.ymax = y;
            }
            if (y < Parent.ymin) {
                Parent.ymin = y;
            }

            //xx and yy are the scaled Xs and Ys used for plotting
            for (int i = 0; i < Xs.length; i++){
                if(Ys[i] != 0 || Xs[i] != 0) {
                    xx[i] = (int) (50 + (((Xs[i] - Parent.xmin) / (Parent.xmax - Parent.xmin)) * (Parent.xdim - 100)));
                    yy[i] = (int) ((Parent.ydim - 50) - (((Ys[i] - Parent.ymin) / (Parent.ymax - Parent.ymin)) * (Parent.ydim - 100)));
                }
            }
            counter++;
        }

        public void addLabel(String label){
            name = label;
        }

        public void setColor(Color c){
            color = c;
        };

        public void paint(Graphics g){
            // Draw Lines
            Color c1 = g.getColor();
            if(color != null){
                g.setColor(color);
            }
            if(name != null){
                g.drawString(name, xx[0],yy[0]);
            }
            if(BarGraph){
                for (int j = 0; j < Xs.length-1; j++)
                {
                    if(Xs[j+1] > 0) {
                        g.drawLine(xx[j],yymin , xx[j], yy[j]);
                    }
                }
            }else {
                for (int j = 0; j < Xs.length - 1; j++) {
                    if (Xs[j + 1] > 0) {
                        g.drawLine(xx[j], yy[j], xx[j + 1], yy[j + 1]);
                    }
                }
            }
            g.setColor(c1);
        }

    }

    public class GamePanel extends JPanel{
        int size = 1000;
        float[][] State;
        boolean Render = true;
        double PixelScale;

        public GamePanel(float[][] state){
            setPreferredSize(new Dimension(size,size));
            setLayout(null);
            State = state;
            PixelScale = size/state.length;
            repaint();
        }

        public void setState(float[][] state){
            State = state;
            if(Render) {
                repaint();
            }
        }

        public void paintComponent(Graphics g){
            super.paintComponent(g);
            Graphics2D g2d = (Graphics2D) g;
            g2d.setColor(Color.BLACK);
            for(int i = (int)PixelScale; i < size; i += PixelScale){
                g2d.drawLine(i,0,i,size);
                g2d.drawLine(0, i, size, i);
            }
            for(int y = 0; y < State.length; y++){
                for(int x = 0; x < State.length; x++){
                    if(State[y][x] == 1){
                        g2d.setColor(Color.RED);
                        g2d.fill(new Ellipse2D.Double(x*PixelScale,y*PixelScale, PixelScale, PixelScale));
                    }else if(State[y][x] == -1){
                        g2d.setColor(Color.GREEN);
                        g2d.fill(new Rectangle2D.Double(x*PixelScale,y*PixelScale,PixelScale,PixelScale));
                    }
                }
            }

        }
    }



}
