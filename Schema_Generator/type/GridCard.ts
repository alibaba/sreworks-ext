/**
 * 表格的位置信息
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

interface Action {
    /**
     * 操作的标签 默认为test
     */
    label: string;
    /**
     * 操作的名称
     */
    name: string;
    /**
     * 操作的图标 默认为setting
     */
    icon: string;
}

interface RowActions {
    /**
     * 操作类型 默认为空字符串
     */
    type: string;
    /**
     * 操作列表 默认为字符串
     */
    layout: string;
    /**
     * 操作列表
     */
    actions: Action[];
}

interface Api {
    /**
     * api的url 默认为空字符串
     */
    url: string;
    /**
     * 是否可分页 默认为false
     */
    paging: boolean;
}

interface Column {
    /**
     * 数据源index 字段名
     */
    dataIndex: string;
    /**
     * 数据源index 标签 ~中文~
     */
    label: string;
}

interface Toolbar {
    /**
     * 工具条的类型 默认为link
     */
    type: string;
    /**
     * 操作提示 默认为操作
     */
    label: string;
    /**
     * 操作列表 默认为空数组
     */
    actionList: any[];
    /**
     * 文档列表 默认为空数组
     */
    docList: any[];
}

interface GridCard {
    /**
     * 网格卡片有几行
     */
    gutter: number;
    /**
     * 网格卡片行上有几个
     */
    column: number;
}

interface GridCard {
    /**
     * 网格卡片的顶部主标题，按照卡片内容总结 ~必须~
     */
    title: string;
    /**
     * 表格的位置信息
     */
    gridPos: GridPos;
    /**
     * 依赖，默认为空数组
     */
    dependParams: any[];
    /**
     * style，默认为null
     */
    style: any;
    /**
     * 输出，默认为空数组
     */
    outputs: any[];
    /**
     * 刷新间隔，默认为null
     */
    refreshInterval: number | null;
    /**
     * 可见性表达式，默认为null
     */
    visibleExp: any | null;
    /**
     * 组件的唯一值，~比如为随机的UUID~
     */
    uniqueKey: string;
    /**
     * flex 默认为 2
     */
    flex: number;
    /**
     * 网格卡片的工具提示
     */
    toolTip: string;
    /**
     * 网格卡片的布局
     */
    grid: GridCard;
    /**
     * 网格卡片的行的操作
     */
    rowActions: RowActions;
    /**
     * 网格卡片的数据源
     */
    api: Api;
    /**
     * 网格卡片的列
     */
    columns: Column[];
    /**
     * 网格卡片的图标 默认为icon
     */
    icon: string;
    /**
     * 网格卡片的工具条
     */
    toolbar: Toolbar;
    /**
     * 布局选择 默认为base平铺布局 advance主次布局
     */
    layoutSelect: "base" | "advance";
    /**
     * 有无卡片封装 默认为default
     * none 无
     */
    hasWrapper: string;
    /**
     * 卡片风格 默认为default
     * advance 高级
     * title_transparent 标题透明
     * transparent 透明
     */
    wrapper: string;
    /**
     * 有无边框 默认为true
     */
    cardBorder: boolean;
    /**
     * 卡片标题颜色 默认为#ffffff
     */
    headerColor: string;
    /**
     * 是否可分页 默认为false
     */
    paging: boolean;
    /**
     * 是否可折叠 默认为false
     */
    foldEnable: boolean;
    /**
     * 是否可隐藏 默认为false
     */
    hiddenEnable: boolean;
}
