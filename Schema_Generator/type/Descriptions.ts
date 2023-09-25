/**
 * 描述列表的位置信息
 */
interface GridPos {
    x: number;
    y: number;
    h: number;
    w: number;
    /**
     * 默认为init
     */
    i: string;
}

interface FormatItem {
    /**
     * 标签名称
     */
    label: string;
    /**
     * 标签的宽度
     */
    span?: number;
    /**
     * 标签的描述
     */
    description?: string;
    /**
     * 标签的数据源
     */
    dataIndex: string;
    /**
     * 标签的链接
     */
    href?: string;
    /**
     * 标签的渲染函数 默认为空
     */
    render?: string;
}

interface DescriptionsConfig {
    title: string;
    gridPos: GridPos;
    dependParams: any[];
    style: null;
    outputs: any[];
    refreshInterval: null;
    visibleExp: null;
    uniqueKey: string;
    formatList: FormatItem[];
    bordered: boolean;
    colon: boolean;
    layout: string;
    labelStyle: any;
    descriptionStyle: { fontSize: number };
    column: {
        xxl: number;
        xl: number;
        lg: number;
        md: number;
        sm: number;
        xs: number;
    };
    minHeight: number;
}

interface DescriptionsData {
    type: string;
    config: DescriptionsConfig;
}