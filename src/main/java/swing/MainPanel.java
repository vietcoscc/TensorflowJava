package swing;

import org.tensorflow.Graph;
import org.tensorflow.Operation;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

public class MainPanel extends JPanel {

    Image imageIcon = new ImageIcon(getClass().getClassLoader().getResource("bienso.jpg")).getImage();

    public MainPanel() throws Exception {
        setLayout(null);
        repaint();
        InputStream modelStream = getClass().getClassLoader().getResourceAsStream("model.pb");
        InputStream labelStream = getClass().getClassLoader().getResourceAsStream("label.txt");
        InputStream imgStream = getClass().getClassLoader().getResourceAsStream("bienso.jpg");


    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;

        g2d.drawImage(imageIcon, 100, 100, 200, 200, null);
    }
}
