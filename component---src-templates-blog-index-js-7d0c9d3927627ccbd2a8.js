(window.webpackJsonp=window.webpackJsonp||[]).push([[9],{"+pmV":function(e,t,r){e.exports={post:"post-module--post--28Mq2",title:"post-module--title--3XBo2",coverImage:"post-module--coverImage--1GM7V",meta:"post-module--meta--3YtjE",tags:"post-module--tags--3RbqF",tag:"post-module--tag--16U9p",darkTheme:"post-module--dark-theme--DXAZf",readMore:"post-module--readMore--3zWML",postContent:"post-module--postContent--1bfnt"}},"5l6m":function(e,t,r){"use strict";var a=r("+uX7"),n=r("m/aQ"),o=r("17+C"),s=r("WD+B"),c=r("gQbX"),i=r("4jnk"),u=r("5Cvy"),l=r("mh2x"),p=Math.max,f=Math.min,m=Math.floor,d=/\$([$&'`]|\d\d?|<[^>]*>)/g,v=/\$([$&'`]|\d\d?)/g;a("replace",2,(function(e,t,r,a){var g=a.REGEXP_REPLACE_SUBSTITUTES_UNDEFINED_CAPTURE,x=a.REPLACE_KEEPS_$0,b=g?"$":"$0";return[function(r,a){var n=i(this),o=null==r?void 0:r[e];return void 0!==o?o.call(r,n,a):t.call(String(n),r,a)},function(e,a){if(!g&&x||"string"==typeof a&&-1===a.indexOf(b)){var o=r(t,e,this,a);if(o.done)return o.value}var i=n(e),m=String(this),d="function"==typeof a;d||(a=String(a));var v=i.global;if(v){var y=i.unicode;i.lastIndex=0}for(var E=[];;){var P=l(i,m);if(null===P)break;if(E.push(P),!v)break;""===String(P[0])&&(i.lastIndex=u(m,s(i.lastIndex),y))}for(var O,j="",I=0,_=0;_<E.length;_++){P=E[_];for(var M=String(P[0]),k=p(f(c(P.index),m.length),0),R=[],w=1;w<P.length;w++)R.push(void 0===(O=P[w])?O:String(O));var N=P.groups;if(d){var S=[M].concat(R,k,m);void 0!==N&&S.push(N);var L=String(a.apply(void 0,S))}else L=h(M,m,k,R,N,a);k>=I&&(j+=m.slice(I,k)+L,I=k+M.length)}return j+m.slice(I)}];function h(e,r,a,n,s,c){var i=a+e.length,u=n.length,l=v;return void 0!==s&&(s=o(s),l=d),t.call(c,l,(function(t,o){var c;switch(o.charAt(0)){case"$":return"$";case"&":return e;case"`":return r.slice(0,a);case"'":return r.slice(i);case"<":c=s[o.slice(1,-1)];break;default:var l=+o;if(0===l)return t;if(l>u){var p=m(l/10);return 0===p?t:p<=u?void 0===n[p-1]?o.charAt(1):n[p-1]+o.charAt(1):t}c=n[l-1]}return void 0===c?"":c}))}}))},"6cYQ":function(e,t,r){"use strict";var a=r("q1tI"),n=r.n(a),o=r("17x9"),s=r.n(o),c=r("Wbzz"),i=r("zHTP"),u=r.n(i),l=function(e){var t=e.nextPath,r=e.previousPath,a=e.nextLabel,o=e.previousLabel;return r||t?n.a.createElement("div",{className:u.a.navigation},r&&n.a.createElement("span",{className:u.a.button},n.a.createElement(c.Link,{to:r},n.a.createElement("span",{className:u.a.iconPrev},"←"),n.a.createElement("span",{className:u.a.buttonText},o))),t&&n.a.createElement("span",{className:u.a.button},n.a.createElement(c.Link,{to:t},n.a.createElement("span",{className:u.a.buttonText},a),n.a.createElement("span",{className:u.a.iconNext},"→")))):null};l.propTypes={nextPath:s.a.string,previousPath:s.a.string,nextLabel:s.a.string,previousLabel:s.a.string},t.a=l},"9yfP":function(e,t,r){"use strict";r.r(t);r("LJRI");var a=r("q1tI"),n=r.n(a),o=r("17x9"),s=r.n(o),c=r("vrFN"),i=r("Bl7J"),u=r("rgsX"),l=r("6cYQ"),p=function(e){var t=e.data,r=e.pageContext,a=r.nextPagePath,o=r.previousPagePath,s=t.allMdx.edges;return n.a.createElement(n.a.Fragment,null,n.a.createElement(c.a,null),n.a.createElement(i.a,{column:!0},s.map((function(e){var t=e.node,r=t.id,a=t.excerpt,o=t.frontmatter,s=o.title,c=o.date,i=o.path,l=o.author,p=o.coverImage,f=o.excerpt,m=o.tags;return n.a.createElement(u.a,{key:r,title:s,date:c,path:i,author:l,coverImage:p,tags:m,excerpt:f||a})})),n.a.createElement(l.a,{previousPath:o,previousLabel:"Newer posts",nextPath:a,nextLabel:"Older posts"})))};p.propTypes={data:s.a.object.isRequired,pageContext:s.a.shape({nextPagePath:s.a.string,previousPagePath:s.a.string})};t.default=p},"A2+M":function(e,t,r){var a=r("X8hv");e.exports={MDXRenderer:a}},Bnag:function(e,t){e.exports=function(){throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")},e.exports.default=e.exports,e.exports.__esModule=!0},EbDI:function(e,t){e.exports=function(e){if("undefined"!=typeof Symbol&&Symbol.iterator in Object(e))return Array.from(e)},e.exports.default=e.exports,e.exports.__esModule=!0},Ijbi:function(e,t,r){var a=r("WkPL");e.exports=function(e){if(Array.isArray(e))return a(e)},e.exports.default=e.exports,e.exports.__esModule=!0},RIqP:function(e,t,r){var a=r("Ijbi"),n=r("EbDI"),o=r("ZhPi"),s=r("Bnag");e.exports=function(e){return a(e)||n(e)||o(e)||s()},e.exports.default=e.exports,e.exports.__esModule=!0},WkPL:function(e,t){e.exports=function(e,t){(null==t||t>e.length)&&(t=e.length);for(var r=0,a=new Array(t);r<t;r++)a[r]=e[r];return a},e.exports.default=e.exports,e.exports.__esModule=!0},X8hv:function(e,t,r){var a=r("sXyB"),n=r("RIqP"),o=r("lSNA"),s=r("8OQS");function c(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,a)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?c(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):c(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var u=r("q1tI"),l=r("7ljp").mdx,p=r("BfwJ").useMDXScope;e.exports=function(e){var t=e.scope,r=e.children,o=s(e,["scope","children"]),c=p(t),f=u.useMemo((function(){if(!r)return null;var e=i({React:u,mdx:l},c),t=Object.keys(e),o=t.map((function(t){return e[t]}));return a(Function,["_fn"].concat(n(t),[""+r])).apply(void 0,[{}].concat(n(o)))}),[r,t]);return u.createElement(f,i({},o))}},ZhPi:function(e,t,r){var a=r("WkPL");e.exports=function(e,t){if(e){if("string"==typeof e)return a(e,t);var r=Object.prototype.toString.call(e).slice(8,-1);return"Object"===r&&e.constructor&&(r=e.constructor.name),"Map"===r||"Set"===r?Array.from(e):"Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)?a(e,t):void 0}},e.exports.default=e.exports,e.exports.__esModule=!0},b48C:function(e,t){e.exports=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Boolean.prototype.valueOf.call(Reflect.construct(Boolean,[],(function(){}))),!0}catch(e){return!1}},e.exports.default=e.exports,e.exports.__esModule=!0},kAaw:function(e,t,r){var a=r("IvzW"),n=r("REpN"),o=r("ZRnM"),s=r("nynY"),c=r("jekk").f,i=r("zpoX").f,u=r("iwAE"),l=r("7npg"),p=r("zJsW"),f=r("+7hJ"),m=r("JhOX"),d=r("E9J1").set,v=r("43WK"),g=r("QD2z")("match"),x=n.RegExp,b=x.prototype,h=/a/g,y=/a/g,E=new x(h)!==h,P=p.UNSUPPORTED_Y;if(a&&o("RegExp",!E||P||m((function(){return y[g]=!1,x(h)!=h||x(y)==y||"/a/i"!=x(h,"i")})))){for(var O=function(e,t){var r,a=this instanceof O,n=u(e),o=void 0===t;if(!a&&n&&e.constructor===O&&o)return e;E?n&&!o&&(e=e.source):e instanceof O&&(o&&(t=l.call(e)),e=e.source),P&&(r=!!t&&t.indexOf("y")>-1)&&(t=t.replace(/y/g,""));var c=s(E?new x(e,t):x(e,t),a?this:b,O);return P&&r&&d(c,{sticky:r}),c},j=function(e){e in O||c(O,e,{configurable:!0,get:function(){return x[e]},set:function(t){x[e]=t}})},I=i(x),_=0;I.length>_;)j(I[_++]);b.constructor=O,O.prototype=b,f(n,"RegExp",O)}v("RegExp")},rgsX:function(e,t,r){"use strict";r("LJRI");var a=r("q1tI"),n=r.n(a),o=r("17x9"),s=r.n(o),c=r("Wbzz"),i=r("9eSz"),u=r.n(i),l=r("A2+M"),p=r("6cYQ"),f=r("zpb6"),m=r("+pmV"),d=r.n(m),v=function(e){var t=e.title,r=e.date,a=e.path,o=e.coverImage,s=e.author,i=e.excerpt,m=e.tags,v=e.body,g=e.previousPost,x=e.nextPost,b=g&&g.frontmatter.path,h=g&&g.frontmatter.title,y=x&&x.frontmatter.path,E=x&&x.frontmatter.title;return n.a.createElement("div",{className:d.a.post},n.a.createElement("div",{className:d.a.postContent},n.a.createElement("h1",{className:d.a.title},i?n.a.createElement(c.Link,{to:a},t):t),n.a.createElement("div",{className:d.a.meta},r," ",s&&n.a.createElement(n.a.Fragment,null,"— Written by ",s),m?n.a.createElement("div",{className:d.a.tags},m.map((function(e){return n.a.createElement(c.Link,{to:"/tag/"+Object(f.toKebabCase)(e)+"/",key:Object(f.toKebabCase)(e)},n.a.createElement("span",{className:d.a.tag},"#",e))}))):null),o&&n.a.createElement(u.a,{fluid:o.childImageSharp.fluid,className:d.a.coverImage}),i?n.a.createElement(n.a.Fragment,null,n.a.createElement("p",null,i),n.a.createElement(c.Link,{to:a,className:d.a.readMore},"Read more →")):n.a.createElement(n.a.Fragment,null,n.a.createElement(l.MDXRenderer,null,v),n.a.createElement(p.a,{previousPath:b,previousLabel:h,nextPath:y,nextLabel:E}))))};v.propTypes={title:s.a.string,date:s.a.string,path:s.a.string,coverImage:s.a.object,author:s.a.string,excerpt:s.a.string,body:s.a.any,tags:s.a.arrayOf(s.a.string),previousPost:s.a.object,nextPost:s.a.object},t.a=v},sXyB:function(e,t,r){var a=r("SksO"),n=r("b48C");function o(t,r,s){return n()?(e.exports=o=Reflect.construct,e.exports.default=e.exports,e.exports.__esModule=!0):(e.exports=o=function(e,t,r){var n=[null];n.push.apply(n,t);var o=new(Function.bind.apply(e,n));return r&&a(o,r.prototype),o},e.exports.default=e.exports,e.exports.__esModule=!0),o.apply(null,arguments)}e.exports=o,e.exports.default=e.exports,e.exports.__esModule=!0},zHTP:function(e,t,r){e.exports={navigation:"navigation-module--navigation--3Zfju",button:"navigation-module--button--28kp3",buttonText:"navigation-module--buttonText--1Xod2",iconNext:"navigation-module--iconNext--3xyJ-",iconPrev:"navigation-module--iconPrev--23mg1"}},zpb6:function(e,t,r){r("5l6m"),r("kAaw"),e.exports.toKebabCase=function(e){return e.replace(new RegExp("(\\s|_|-)+","gmi"),"-")}}}]);
//# sourceMappingURL=component---src-templates-blog-index-js-7d0c9d3927627ccbd2a8.js.map