package swing;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.nio.Buffer;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Vector;

public class MainPanel extends JPanel {

    private Graphics2D g2d;
    BufferedImage hugeImage;
    BufferedImage smallImage;
    List<Classifier.Recognition> results;

    public MainPanel() throws Exception {
        setLayout(null);
        repaint();
        String resourcePath = getClass().getClassLoader().getResource("").getPath();
        Path modelPath = Paths.get(resourcePath, "ssd_mobilenet_v1_android_export.pb");
        byte modelStream[] = Files.readAllBytes(modelPath);
        Path labelPath = Paths.get(resourcePath, "coco_labels_list.txt");
        Vector<String> labelStream = new Vector<>(Files.readAllLines(labelPath, Charset.forName("UTF-8")));
        this.hugeImage = ImageIO.read(getClass().getResource("/bienso2.jpg"));
        this.smallImage = resizeImage(hugeImage, hugeImage.getType());

        TensorFlowObjectDetectionAPIModel detector = (TensorFlowObjectDetectionAPIModel)
                TensorFlowObjectDetectionAPIModel.create(modelStream, labelStream, 300);
        byte[] pixels = ((DataBufferByte) smallImage.getRaster().getDataBuffer()).getData();
        System.out.println(pixels.length);
        results = detector.recognizeImage(pixels);
        System.out.println(results);
    }

    private static BufferedImage resizeImage(BufferedImage originalImage, int type) {
        BufferedImage resizedImage = new BufferedImage(300, 300, type);
        Graphics2D g = resizedImage.createGraphics();
        g.drawImage(originalImage, 0, 0, 300, 300, null);
        g.dispose();
        return resizedImage;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        this.g2d = (Graphics2D) g;

        g2d.drawImage(smallImage, 0, 0, null);
        g2d.drawImage(hugeImage, smallImage.getWidth() + 1, 0, null);

        g2d.setStroke(new BasicStroke(5));
        g2d.setColor(Color.RED);
        System.out.println(smallImage.getWidth() + "," + smallImage.getHeight());
        System.out.println(hugeImage.getWidth() + "," + hugeImage.getHeight());

//        BasicStroke basicStroke= new BasicStroke();
        for (int i = 0; i < results.size(); i++) {
            Classifier.Recognition result = results.get(i);
            if (result.getConfidence() >= 0.99) {
                RectF location = result.getLocation();
                Rectangle rectangle = rectF2Rectangle(location);
                g2d.draw(rectangle);

                float w = hugeImage.getWidth() * 1.0f / smallImage.getWidth();
                float h = hugeImage.getHeight() * 1.0f / smallImage.getHeight();

                int x = (int) (rectangle.x * w);
                int y = (int) (rectangle.y * h);
                int width = (int) (rectangle.width * w);
                int height = (int) (rectangle.height * h);

                Rectangle rectangle1 = new Rectangle(x+smallImage.getWidth()+1, y, width, height);
                g2d.draw(rectangle1);
            }
        }
    }

    private Rectangle rectF2Rectangle(RectF rectF) {
        return new Rectangle((int) rectF.left, (int) rectF.top, (int) rectF.right - (int) rectF.left, (int) rectF.bottom - (int) rectF.top);
    }

}
