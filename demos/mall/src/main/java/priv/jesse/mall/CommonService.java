package priv.jesse.mall;

import java.util.Hashtable;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

import com.alibaba.fastjson.JSONObject;

import org.springframework.stereotype.Service;

@Service
public class CommonService {

    public static Map<String, AtomicLong> metrics = new Hashtable<>();

    public static JSONObject metricJsonObject() {
        JSONObject ret = new JSONObject();
        for (String key : metrics.keySet()) {
            ret.put(key, metrics.get(key).longValue());
        }
        return ret;
    }

    public static void add(String url) {
        if (!metrics.containsKey(url)) {
            metrics.put(url, new AtomicLong(0));
        }
        if (!metrics.containsKey("total")) {
            metrics.put("total", new AtomicLong(0));
        }
        metrics.get(url).addAndGet(1);
        metrics.get("total").addAndGet(1);
    }

}
