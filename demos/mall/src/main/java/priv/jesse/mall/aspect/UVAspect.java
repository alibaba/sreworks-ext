package priv.jesse.mall.aspect;

import java.util.concurrent.atomic.AtomicLong;

import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.annotation.After;
import org.aspectj.lang.annotation.Aspect;
import org.springframework.stereotype.Component;

/**
 * 计算方法调用时间切面
 *
 * @author yangjinghua
 */
@Aspect
@Component
public class UVAspect {

    public static AtomicLong total = new AtomicLong(0);

    @After("@annotation(UV)")
    public void afterMethod(JoinPoint joinPoint) {
        System.out.println(total.addAndGet(1));
    }

}

