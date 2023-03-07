//package priv.jesse.mall.filter;
//
//import java.io.IOException;
//import java.util.concurrent.atomic.AtomicLong;
//
//import javax.servlet.Filter;
//import javax.servlet.FilterChain;
//import javax.servlet.FilterConfig;
//import javax.servlet.ServletException;
//import javax.servlet.ServletRequest;
//import javax.servlet.ServletResponse;
//import javax.servlet.annotation.WebFilter;
//import javax.servlet.http.HttpServletRequest;
//
//import com.alibaba.fastjson.JSONObject;
//
//import org.slf4j.Logger;
//import org.slf4j.LoggerFactory;
//import priv.jesse.mall.CommonService;
//import priv.jesse.mall.Requests;
//import sun.misc.Request;
//
///**
// * @author hfb
// * @date 2017/9/18
// */
//@WebFilter
//public class MetricFilter implements Filter {
//
//    public MetricFilter() {
//    }
//
//    private static final Logger LOGGER = LoggerFactory.getLogger(MetricFilter.class);
//
//    @Override
//    public void init(FilterConfig filterConfig) {
//    }
//
//    @Override
//    public void doFilter(ServletRequest req, ServletResponse res, FilterChain chain)
//        throws IOException, ServletException {
//        String url = ((HttpServletRequest)req).getServletPath();
//        CommonService.add(url);
//        new Requests("http://127.0.0.1:10080").postJson(CommonService.metricJsonObject()).post();
//        chain.doFilter(req, res);
//    }
//
//    @Override
//    public void destroy() {
//
//    }
//
//}
