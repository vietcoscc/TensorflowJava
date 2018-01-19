package swing;

import org.tensorflow.*;


import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class TensorFlowInferenceJavaInterface {
    private final Graph g;
    private final Session sess;
    private Session.Runner runner;
    private List<String> feedNames = new ArrayList();
    private List<Tensor> feedTensors = new ArrayList();
    private List<String> fetchNames = new ArrayList();
    private List<Tensor> fetchTensors = new ArrayList();


    public TensorFlowInferenceJavaInterface(byte[] graphDef) {
        this.g = new Graph();
        this.sess = new Session(this.g);
        this.runner = this.sess.runner();

        try {
            loadGraph(graphDef, this.g);
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    private void loadGraph(byte[] graphDef, Graph g) throws IOException {
        try {
            g.importGraphDef(graphDef);
        } catch (IllegalArgumentException var7) {
            throw new IOException("Not a valid TensorFlow Graph serialization: " + var7.getMessage());
        }
    }

    public void feed(String inputName, byte[] src, long... dims) {
        this.addFeed(inputName, Tensor.create(DataType.UINT8, dims, ByteBuffer.wrap(src)));
    }

    public Graph graph() {
        return this.g;
    }

    private void addFeed(String inputName, Tensor t) {
        TensorFlowInferenceJavaInterface.TensorId tid = TensorFlowInferenceJavaInterface.TensorId.parse(inputName);
        this.runner.feed(tid.name, tid.outputIndex, t);
        this.feedNames.add(inputName);
        this.feedTensors.add(t);
    }

    private static class TensorId {
        String name;
        int outputIndex;

        private TensorId() {
        }

        public static TensorFlowInferenceJavaInterface.TensorId parse(String name) {
            TensorFlowInferenceJavaInterface.TensorId tid = new TensorFlowInferenceJavaInterface.TensorId();
            int colonIndex = name.lastIndexOf(58);
            if (colonIndex < 0) {
                tid.outputIndex = 0;
                tid.name = name;
                return tid;
            } else {
                try {
                    tid.outputIndex = Integer.parseInt(name.substring(colonIndex + 1));
                    tid.name = name.substring(0, colonIndex);
                } catch (NumberFormatException var4) {
                    tid.outputIndex = 0;
                    tid.name = name;
                }

                return tid;
            }
        }
    }

    public void run(String[] outputNames) {
        this.closeFetches();
        String[] var3 = outputNames;
        int var4 = outputNames.length;

        for (int var5 = 0; var5 < var4; ++var5) {
            String o = var3[var5];
            this.fetchNames.add(o);
            TensorFlowInferenceJavaInterface.TensorId tid = TensorFlowInferenceJavaInterface.TensorId.parse(o);
            this.runner.fetch(tid.name, tid.outputIndex);
        }

        try {
            this.fetchTensors = this.runner.run();
        } catch (RuntimeException var11) {
            throw var11;
        } finally {
            this.closeFeeds();
            this.runner = this.sess.runner();
        }
    }
    private void closeFeeds() {
        Iterator var1 = this.feedTensors.iterator();

        while(var1.hasNext()) {
            Tensor t = (Tensor)var1.next();
            t.close();
        }

        this.feedTensors.clear();
        this.feedNames.clear();
    }
    private void closeFetches() {
        Iterator var1 = this.fetchTensors.iterator();

        while (var1.hasNext()) {
            Tensor t = (Tensor) var1.next();
            t.close();
        }

        this.fetchTensors.clear();
        this.fetchNames.clear();
    }

    private Tensor getTensor(String outputName) {
        int i = 0;

        for (Iterator var3 = this.fetchNames.iterator(); var3.hasNext(); ++i) {
            String n = (String) var3.next();
            if (n.equals(outputName)) {
                return this.fetchTensors.get(i);
            }
        }

        throw new RuntimeException("Node '" + outputName + "' was not provided to run(), so it cannot be read");
    }

    public void fetch(String outputName, float[] dst) {
        this.fetch(outputName, FloatBuffer.wrap(dst));
    }

    public void fetch(String outputName, FloatBuffer dst) {
        this.getTensor(outputName).writeTo(dst);
    }
}
