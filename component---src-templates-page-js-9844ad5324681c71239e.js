(window.webpackJsonp=window.webpackJsonp||[]).push([[10],{"+pmV":function(e,t,r){e.exports={post:"post-module--post--28Mq2",title:"post-module--title--3XBo2",coverImage:"post-module--coverImage--1GM7V",meta:"post-module--meta--3YtjE",tags:"post-module--tags--3RbqF",tag:"post-module--tag--16U9p",darkTheme:"post-module--dark-theme--DXAZf",readMore:"post-module--readMore--3zWML",postContent:"post-module--postContent--1bfnt"}},"5l6m":function(e,t,r){"use strict";var n=r("+uX7"),a=r("m/aQ"),o=r("17+C"),s=r("WD+B"),c=r("gQbX"),i=r("4jnk"),u=r("5Cvy"),l=r("mh2x"),p=Math.max,f=Math.min,m=Math.floor,d=/\$([$&'`]|\d\d?|<[^>]*>)/g,v=/\$([$&'`]|\d\d?)/g;n("replace",2,(function(e,t,r,n){var g=n.REGEXP_REPLACE_SUBSTITUTES_UNDEFINED_CAPTURE,x=n.REPLACE_KEEPS_$0,b=g?"$":"$0";return[function(r,n){var a=i(this),o=null==r?void 0:r[e];return void 0!==o?o.call(r,a,n):t.call(String(a),r,n)},function(e,n){if(!g&&x||"string"==typeof n&&-1===n.indexOf(b)){var o=r(t,e,this,n);if(o.done)return o.value}var i=a(e),m=String(this),d="function"==typeof n;d||(n=String(n));var v=i.global;if(v){var y=i.unicode;i.lastIndex=0}for(var E=[];;){var P=l(i,m);if(null===P)break;if(E.push(P),!v)break;""===String(P[0])&&(i.lastIndex=u(m,s(i.lastIndex),y))}for(var j,O="",_=0,I=0;I<E.length;I++){P=E[I];for(var k=String(P[0]),M=p(f(c(P.index),m.length),0),R=[],w=1;w<P.length;w++)R.push(void 0===(j=P[w])?j:String(j));var S=P.groups;if(d){var N=[k].concat(R,M,m);void 0!==S&&N.push(S);var A=String(n.apply(void 0,N))}else A=h(k,m,M,R,S,n);M>=_&&(O+=m.slice(_,M)+A,_=M+k.length)}return O+m.slice(_)}];function h(e,r,n,a,s,c){var i=n+e.length,u=a.length,l=v;return void 0!==s&&(s=o(s),l=d),t.call(c,l,(function(t,o){var c;switch(o.charAt(0)){case"$":return"$";case"&":return e;case"`":return r.slice(0,n);case"'":return r.slice(i);case"<":c=s[o.slice(1,-1)];break;default:var l=+o;if(0===l)return t;if(l>u){var p=m(l/10);return 0===p?t:p<=u?void 0===a[p-1]?o.charAt(1):a[p-1]+o.charAt(1):t}c=a[l-1]}return void 0===c?"":c}))}}))},"6cYQ":function(e,t,r){"use strict";var n=r("q1tI"),a=r.n(n),o=r("17x9"),s=r.n(o),c=r("Wbzz"),i=r("zHTP"),u=r.n(i),l=function(e){var t=e.nextPath,r=e.previousPath,n=e.nextLabel,o=e.previousLabel;return r||t?a.a.createElement("div",{className:u.a.navigation},r&&a.a.createElement("span",{className:u.a.button},a.a.createElement(c.Link,{to:r},a.a.createElement("span",{className:u.a.iconPrev},"←"),a.a.createElement("span",{className:u.a.buttonText},o))),t&&a.a.createElement("span",{className:u.a.button},a.a.createElement(c.Link,{to:t},a.a.createElement("span",{className:u.a.buttonText},n),a.a.createElement("span",{className:u.a.iconNext},"→")))):null};l.propTypes={nextPath:s.a.string,previousPath:s.a.string,nextLabel:s.a.string,previousLabel:s.a.string},t.a=l},"A2+M":function(e,t,r){var n=r("X8hv");e.exports={MDXRenderer:n}},Bnag:function(e,t){e.exports=function(){throw new TypeError("Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.")},e.exports.default=e.exports,e.exports.__esModule=!0},EbDI:function(e,t){e.exports=function(e){if("undefined"!=typeof Symbol&&Symbol.iterator in Object(e))return Array.from(e)},e.exports.default=e.exports,e.exports.__esModule=!0},Ijbi:function(e,t,r){var n=r("WkPL");e.exports=function(e){if(Array.isArray(e))return n(e)},e.exports.default=e.exports,e.exports.__esModule=!0},RIqP:function(e,t,r){var n=r("Ijbi"),a=r("EbDI"),o=r("ZhPi"),s=r("Bnag");e.exports=function(e){return n(e)||a(e)||o(e)||s()},e.exports.default=e.exports,e.exports.__esModule=!0},WkPL:function(e,t){e.exports=function(e,t){(null==t||t>e.length)&&(t=e.length);for(var r=0,n=new Array(t);r<t;r++)n[r]=e[r];return n},e.exports.default=e.exports,e.exports.__esModule=!0},X8hv:function(e,t,r){var n=r("sXyB"),a=r("RIqP"),o=r("lSNA"),s=r("8OQS");function c(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function i(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?c(Object(r),!0).forEach((function(t){o(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):c(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}var u=r("q1tI"),l=r("7ljp").mdx,p=r("BfwJ").useMDXScope;e.exports=function(e){var t=e.scope,r=e.children,o=s(e,["scope","children"]),c=p(t),f=u.useMemo((function(){if(!r)return null;var e=i({React:u,mdx:l},c),t=Object.keys(e),o=t.map((function(t){return e[t]}));return n(Function,["_fn"].concat(a(t),[""+r])).apply(void 0,[{}].concat(a(o)))}),[r,t]);return u.createElement(f,i({},o))}},ZhPi:function(e,t,r){var n=r("WkPL");e.exports=function(e,t){if(e){if("string"==typeof e)return n(e,t);var r=Object.prototype.toString.call(e).slice(8,-1);return"Object"===r&&e.constructor&&(r=e.constructor.name),"Map"===r||"Set"===r?Array.from(e):"Arguments"===r||/^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(r)?n(e,t):void 0}},e.exports.default=e.exports,e.exports.__esModule=!0},b48C:function(e,t){e.exports=function(){if("undefined"==typeof Reflect||!Reflect.construct)return!1;if(Reflect.construct.sham)return!1;if("function"==typeof Proxy)return!0;try{return Boolean.prototype.valueOf.call(Reflect.construct(Boolean,[],(function(){}))),!0}catch(e){return!1}},e.exports.default=e.exports,e.exports.__esModule=!0},kAaw:function(e,t,r){var n=r("IvzW"),a=r("REpN"),o=r("ZRnM"),s=r("nynY"),c=r("jekk").f,i=r("zpoX").f,u=r("iwAE"),l=r("7npg"),p=r("zJsW"),f=r("+7hJ"),m=r("JhOX"),d=r("E9J1").set,v=r("43WK"),g=r("QD2z")("match"),x=a.RegExp,b=x.prototype,h=/a/g,y=/a/g,E=new x(h)!==h,P=p.UNSUPPORTED_Y;if(n&&o("RegExp",!E||P||m((function(){return y[g]=!1,x(h)!=h||x(y)==y||"/a/i"!=x(h,"i")})))){for(var j=function(e,t){var r,n=this instanceof j,a=u(e),o=void 0===t;if(!n&&a&&e.constructor===j&&o)return e;E?a&&!o&&(e=e.source):e instanceof j&&(o&&(t=l.call(e)),e=e.source),P&&(r=!!t&&t.indexOf("y")>-1)&&(t=t.replace(/y/g,""));var c=s(E?new x(e,t):x(e,t),n?this:b,j);return P&&r&&d(c,{sticky:r}),c},O=function(e){e in j||c(j,e,{configurable:!0,get:function(){return x[e]},set:function(t){x[e]=t}})},_=i(x),I=0;_.length>I;)O(_[I++]);b.constructor=j,j.prototype=b,f(a,"RegExp",j)}v("RegExp")},rgsX:function(e,t,r){"use strict";r("LJRI");var n=r("q1tI"),a=r.n(n),o=r("17x9"),s=r.n(o),c=r("Wbzz"),i=r("9eSz"),u=r.n(i),l=r("A2+M"),p=r("6cYQ"),f=r("zpb6"),m=r("+pmV"),d=r.n(m),v=function(e){var t=e.title,r=e.date,n=e.path,o=e.coverImage,s=e.author,i=e.excerpt,m=e.tags,v=e.body,g=e.previousPost,x=e.nextPost,b=g&&g.frontmatter.path,h=g&&g.frontmatter.title,y=x&&x.frontmatter.path,E=x&&x.frontmatter.title;return a.a.createElement("div",{className:d.a.post},a.a.createElement("div",{className:d.a.postContent},a.a.createElement("h1",{className:d.a.title},i?a.a.createElement(c.Link,{to:n},t):t),a.a.createElement("div",{className:d.a.meta},r," ",s&&a.a.createElement(a.a.Fragment,null,"— Written by ",s),m?a.a.createElement("div",{className:d.a.tags},m.map((function(e){return a.a.createElement(c.Link,{to:"/tag/"+Object(f.toKebabCase)(e)+"/",key:Object(f.toKebabCase)(e)},a.a.createElement("span",{className:d.a.tag},"#",e))}))):null),o&&a.a.createElement(u.a,{fluid:o.childImageSharp.fluid,className:d.a.coverImage}),i?a.a.createElement(a.a.Fragment,null,a.a.createElement("p",null,i),a.a.createElement(c.Link,{to:n,className:d.a.readMore},"Read more →")):a.a.createElement(a.a.Fragment,null,a.a.createElement(l.MDXRenderer,null,v),a.a.createElement(p.a,{previousPath:b,previousLabel:h,nextPath:y,nextLabel:E}))))};v.propTypes={title:s.a.string,date:s.a.string,path:s.a.string,coverImage:s.a.object,author:s.a.string,excerpt:s.a.string,body:s.a.any,tags:s.a.arrayOf(s.a.string),previousPost:s.a.object,nextPost:s.a.object},t.a=v},sXyB:function(e,t,r){var n=r("SksO"),a=r("b48C");function o(t,r,s){return a()?(e.exports=o=Reflect.construct,e.exports.default=e.exports,e.exports.__esModule=!0):(e.exports=o=function(e,t,r){var a=[null];a.push.apply(a,t);var o=new(Function.bind.apply(e,a));return r&&n(o,r.prototype),o},e.exports.default=e.exports,e.exports.__esModule=!0),o.apply(null,arguments)}e.exports=o,e.exports.default=e.exports,e.exports.__esModule=!0},sweJ:function(e,t,r){"use strict";r.r(t);var n=r("q1tI"),a=r.n(n),o=r("17x9"),s=r.n(o),c=r("vrFN"),i=r("Bl7J"),u=r("rgsX"),l=function(e){var t=e.data.mdx,r=t.frontmatter,n=r.title,o=r.path,s=r.coverImage,l=r.excerpt,p=r.tags,f=t.excerpt,m=t.id,d=t.body;return a.a.createElement(i.a,null,a.a.createElement(c.a,{title:n,description:l||f}),a.a.createElement(u.a,{key:m,title:n,path:o,coverImage:s,body:d,tags:p}))};t.default=l,l.propTypes={data:s.a.object.isRequired,pageContext:s.a.shape({next:s.a.object,previous:s.a.object})}},zHTP:function(e,t,r){e.exports={navigation:"navigation-module--navigation--3Zfju",button:"navigation-module--button--28kp3",buttonText:"navigation-module--buttonText--1Xod2",iconNext:"navigation-module--iconNext--3xyJ-",iconPrev:"navigation-module--iconPrev--23mg1"}},zpb6:function(e,t,r){r("5l6m"),r("kAaw"),e.exports.toKebabCase=function(e){return e.replace(new RegExp("(\\s|_|-)+","gmi"),"-")}}}]);
//# sourceMappingURL=component---src-templates-page-js-9844ad5324681c71239e.js.map