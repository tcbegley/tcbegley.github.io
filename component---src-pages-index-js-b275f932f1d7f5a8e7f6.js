/*! For license information please see component---src-pages-index-js-b275f932f1d7f5a8e7f6.js.LICENSE.txt */
(window.webpackJsonp=window.webpackJsonp||[]).push([[7],{"7O5W":function(n,t,e){"use strict";(function(n,r){function a(n){return(a="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(n){return typeof n}:function(n){return n&&"function"==typeof Symbol&&n.constructor===Symbol&&n!==Symbol.prototype?"symbol":typeof n})(n)}function i(n,t){for(var e=0;e<t.length;e++){var r=t[e];r.enumerable=r.enumerable||!1,r.configurable=!0,"value"in r&&(r.writable=!0),Object.defineProperty(n,r.key,r)}}function o(n,t,e){return t in n?Object.defineProperty(n,t,{value:e,enumerable:!0,configurable:!0,writable:!0}):n[t]=e,n}function c(n){for(var t=1;t<arguments.length;t++){var e=null!=arguments[t]?arguments[t]:{},r=Object.keys(e);"function"==typeof Object.getOwnPropertySymbols&&(r=r.concat(Object.getOwnPropertySymbols(e).filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable})))),r.forEach((function(t){o(n,t,e[t])}))}return n}function s(n,t){return function(n){if(Array.isArray(n))return n}(n)||function(n,t){var e=[],r=!0,a=!1,i=void 0;try{for(var o,c=n[Symbol.iterator]();!(r=(o=c.next()).done)&&(e.push(o.value),!t||e.length!==t);r=!0);}catch(s){a=!0,i=s}finally{try{r||null==c.return||c.return()}finally{if(a)throw i}}return e}(n,t)||function(){throw new TypeError("Invalid attempt to destructure non-iterable instance")}()}e.d(t,"b",(function(){return En})),e.d(t,"a",(function(){return _})),e.d(t,"c",(function(){return _n}));var l=function(){},f={},u={},m={mark:l,measure:l};try{"undefined"!=typeof window&&(f=window),"undefined"!=typeof document&&(u=document),"undefined"!=typeof MutationObserver&&MutationObserver,"undefined"!=typeof performance&&(m=performance)}catch(Tn){}var d=(f.navigator||{}).userAgent,p=void 0===d?"":d,h=f,g=u,y=m,b=(h.document,!!g.documentElement&&!!g.head&&"function"==typeof g.addEventListener&&"function"==typeof g.createElement),v=(~p.indexOf("MSIE")||p.indexOf("Trident/"),function(){try{}catch(Tn){return!1}}(),[1,2,3,4,5,6,7,8,9,10]),w=v.concat([11,12,13,14,15,16,17,18,19,20]),x={GROUP:"group",SWAP_OPACITY:"swap-opacity",PRIMARY:"primary",SECONDARY:"secondary"},k=(["xs","sm","lg","fw","ul","li","border","pull-left","pull-right","spin","pulse","rotate-90","rotate-180","rotate-270","flip-horizontal","flip-vertical","flip-both","stack","stack-1x","stack-2x","inverse","layers","layers-text","layers-counter",x.GROUP,x.SWAP_OPACITY,x.PRIMARY,x.SECONDARY].concat(v.map((function(n){return"".concat(n,"x")}))).concat(w.map((function(n){return"w-".concat(n)}))),h.FontAwesomeConfig||{});if(g&&"function"==typeof g.querySelector){[["data-family-prefix","familyPrefix"],["data-replacement-class","replacementClass"],["data-auto-replace-svg","autoReplaceSvg"],["data-auto-add-css","autoAddCss"],["data-auto-a11y","autoA11y"],["data-search-pseudo-elements","searchPseudoElements"],["data-observe-mutations","observeMutations"],["data-mutate-approach","mutateApproach"],["data-keep-original-source","keepOriginalSource"],["data-measure-performance","measurePerformance"],["data-show-missing-icons","showMissingIcons"]].forEach((function(n){var t=s(n,2),e=t[0],r=t[1],a=function(n){return""===n||"false"!==n&&("true"===n||n)}(function(n){var t=g.querySelector("script["+n+"]");if(t)return t.getAttribute(n)}(e));null!=a&&(k[r]=a)}))}var O=c({},{familyPrefix:"fa",replacementClass:"svg-inline--fa",autoReplaceSvg:!0,autoAddCss:!0,autoA11y:!0,searchPseudoElements:!1,observeMutations:!0,mutateApproach:"async",keepOriginalSource:!0,measurePerformance:!1,showMissingIcons:!0},k);O.autoReplaceSvg||(O.observeMutations=!1);var _=c({},O);h.FontAwesomeConfig=_;var E=h||{};E.___FONT_AWESOME___||(E.___FONT_AWESOME___={}),E.___FONT_AWESOME___.styles||(E.___FONT_AWESOME___.styles={}),E.___FONT_AWESOME___.hooks||(E.___FONT_AWESOME___.hooks={}),E.___FONT_AWESOME___.shims||(E.___FONT_AWESOME___.shims=[]);var T=E.___FONT_AWESOME___,I=[];b&&((g.documentElement.doScroll?/^loaded|^c/:/^loaded|^i|^c/).test(g.readyState)||g.addEventListener("DOMContentLoaded",(function n(){g.removeEventListener("DOMContentLoaded",n),1,I.map((function(n){return n()}))})));var j,A=function(){},C=void 0!==n&&void 0!==n.process&&"function"==typeof n.process.emit,P=void 0===r?setTimeout:r,M=[];function z(){for(var n=0;n<M.length;n++)M[n][0](M[n][1]);M=[],j=!1}function S(n,t){M.push([n,t]),j||(j=!0,P(z,0))}function N(n){var t=n.owner,e=t._state,r=t._data,a=n[e],i=n.then;if("function"==typeof a){e="fulfilled";try{r=a(r)}catch(Tn){F(i,Tn)}}L(i,r)||("fulfilled"===e&&R(i,r),"rejected"===e&&F(i,r))}function L(n,t){var e;try{if(n===t)throw new TypeError("A promises callback cannot return that same promise.");if(t&&("function"==typeof t||"object"===a(t))){var r=t.then;if("function"==typeof r)return r.call(t,(function(r){e||(e=!0,t===r?W(n,r):R(n,r))}),(function(t){e||(e=!0,F(n,t))})),!0}}catch(Tn){return e||F(n,Tn),!0}return!1}function R(n,t){n!==t&&L(n,t)||W(n,t)}function W(n,t){"pending"===n._state&&(n._state="settled",n._data=t,S(D,n))}function F(n,t){"pending"===n._state&&(n._state="settled",n._data=t,S(B,n))}function Y(n){n._then=n._then.forEach(N)}function D(n){n._state="fulfilled",Y(n)}function B(t){t._state="rejected",Y(t),!t._handled&&C&&n.process.emit("unhandledRejection",t._data,t)}function X(t){n.process.emit("rejectionHandled",t)}function U(n){if("function"!=typeof n)throw new TypeError("Promise resolver "+n+" is not a function");if(this instanceof U==!1)throw new TypeError("Failed to construct 'Promise': Please use the 'new' operator, this object constructor cannot be called as a function.");this._then=[],function(n,t){function e(n){F(t,n)}try{n((function(n){R(t,n)}),e)}catch(Tn){e(Tn)}}(n,this)}U.prototype={constructor:U,_state:"pending",_then:null,_data:void 0,_handled:!1,then:function(n,t){var e={owner:this,then:new this.constructor(A),fulfilled:n,rejected:t};return!t&&!n||this._handled||(this._handled=!0,"rejected"===this._state&&C&&S(X,this)),"fulfilled"===this._state||"rejected"===this._state?S(N,e):this._then.push(e),e.then},catch:function(n){return this.then(null,n)}},U.all=function(n){if(!Array.isArray(n))throw new TypeError("You must pass an array to Promise.all().");return new U((function(t,e){var r=[],a=0;function i(n){return a++,function(e){r[n]=e,--a||t(r)}}for(var o,c=0;c<n.length;c++)(o=n[c])&&"function"==typeof o.then?o.then(i(c),e):r[c]=o;a||t(r)}))},U.race=function(n){if(!Array.isArray(n))throw new TypeError("You must pass an array to Promise.race().");return new U((function(t,e){for(var r,a=0;a<n.length;a++)(r=n[a])&&"function"==typeof r.then?r.then(t,e):t(r)}))},U.resolve=function(n){return n&&"object"===a(n)&&n.constructor===U?n:new U((function(t){t(n)}))},U.reject=function(n){return new U((function(t,e){e(n)}))};var H={size:16,x:0,y:0,rotate:0,flipX:!1,flipY:!1};function q(n){if(n&&b){var t=g.createElement("style");t.setAttribute("type","text/css"),t.innerHTML=n;for(var e=g.head.childNodes,r=null,a=e.length-1;a>-1;a--){var i=e[a],o=(i.tagName||"").toUpperCase();["STYLE","LINK"].indexOf(o)>-1&&(r=i)}return g.head.insertBefore(t,r),n}}function J(){for(var n=12,t="";n-- >0;)t+="0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"[62*Math.random()|0];return t}function K(n){return"".concat(n).replace(/&/g,"&amp;").replace(/"/g,"&quot;").replace(/'/g,"&#39;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}function V(n){return Object.keys(n||{}).reduce((function(t,e){return t+"".concat(e,": ").concat(n[e],";")}),"")}function G(n){return n.size!==H.size||n.x!==H.x||n.y!==H.y||n.rotate!==H.rotate||n.flipX||n.flipY}function Z(n){var t=n.transform,e=n.containerWidth,r=n.iconWidth,a={transform:"translate(".concat(e/2," 256)")},i="translate(".concat(32*t.x,", ").concat(32*t.y,") "),o="scale(".concat(t.size/16*(t.flipX?-1:1),", ").concat(t.size/16*(t.flipY?-1:1),") "),c="rotate(".concat(t.rotate," 0 0)");return{outer:a,inner:{transform:"".concat(i," ").concat(o," ").concat(c)},path:{transform:"translate(".concat(r/2*-1," -256)")}}}var $={x:0,y:0,width:"100%",height:"100%"};function Q(n){var t=!(arguments.length>1&&void 0!==arguments[1])||arguments[1];return n.attributes&&(n.attributes.fill||t)&&(n.attributes.fill="black"),n}function nn(n){var t=n.icons,e=t.main,r=t.mask,a=n.prefix,i=n.iconName,o=n.transform,s=n.symbol,l=n.title,f=n.maskId,u=n.titleId,m=n.extra,d=n.watchable,p=void 0!==d&&d,h=r.found?r:e,g=h.width,y=h.height,b="fak"===a,v=b?"":"fa-w-".concat(Math.ceil(g/y*16)),w=[_.replacementClass,i?"".concat(_.familyPrefix,"-").concat(i):"",v].filter((function(n){return-1===m.classes.indexOf(n)})).filter((function(n){return""!==n||!!n})).concat(m.classes).join(" "),x={children:[],attributes:c({},m.attributes,{"data-prefix":a,"data-icon":i,class:w,role:m.attributes.role||"img",xmlns:"http://www.w3.org/2000/svg",viewBox:"0 0 ".concat(g," ").concat(y)})},k=b&&!~m.classes.indexOf("fa-fw")?{width:"".concat(g/y*16*.0625,"em")}:{};p&&(x.attributes["data-fa-i2svg"]=""),l&&x.children.push({tag:"title",attributes:{id:x.attributes["aria-labelledby"]||"title-".concat(u||J())},children:[l]});var O=c({},x,{prefix:a,iconName:i,main:e,mask:r,maskId:f,transform:o,symbol:s,styles:c({},k,m.styles)}),E=r.found&&e.found?function(n){var t,e=n.children,r=n.attributes,a=n.main,i=n.mask,o=n.maskId,s=n.transform,l=a.width,f=a.icon,u=i.width,m=i.icon,d=Z({transform:s,containerWidth:u,iconWidth:l}),p={tag:"rect",attributes:c({},$,{fill:"white"})},h=f.children?{children:f.children.map(Q)}:{},g={tag:"g",attributes:c({},d.inner),children:[Q(c({tag:f.tag,attributes:c({},f.attributes,d.path)},h))]},y={tag:"g",attributes:c({},d.outer),children:[g]},b="mask-".concat(o||J()),v="clip-".concat(o||J()),w={tag:"mask",attributes:c({},$,{id:b,maskUnits:"userSpaceOnUse",maskContentUnits:"userSpaceOnUse"}),children:[p,y]},x={tag:"defs",children:[{tag:"clipPath",attributes:{id:v},children:(t=m,"g"===t.tag?t.children:[t])},w]};return e.push(x,{tag:"rect",attributes:c({fill:"currentColor","clip-path":"url(#".concat(v,")"),mask:"url(#".concat(b,")")},$)}),{children:e,attributes:r}}(O):function(n){var t=n.children,e=n.attributes,r=n.main,a=n.transform,i=V(n.styles);if(i.length>0&&(e.style=i),G(a)){var o=Z({transform:a,containerWidth:r.width,iconWidth:r.width});t.push({tag:"g",attributes:c({},o.outer),children:[{tag:"g",attributes:c({},o.inner),children:[{tag:r.icon.tag,children:r.icon.children,attributes:c({},r.icon.attributes,o.path)}]}]})}else t.push(r.icon);return{children:t,attributes:e}}(O),T=E.children,I=E.attributes;return O.children=T,O.attributes=I,s?function(n){var t=n.prefix,e=n.iconName,r=n.children,a=n.attributes,i=n.symbol;return[{tag:"svg",attributes:{style:"display: none;"},children:[{tag:"symbol",attributes:c({},a,{id:!0===i?"".concat(t,"-").concat(_.familyPrefix,"-").concat(e):i}),children:r}]}]}(O):function(n){var t=n.children,e=n.main,r=n.mask,a=n.attributes,i=n.styles,o=n.transform;if(G(o)&&e.found&&!r.found){var s={x:e.width/e.height/2,y:.5};a.style=V(c({},i,{"transform-origin":"".concat(s.x+o.x/16,"em ").concat(s.y+o.y/16,"em")}))}return[{tag:"svg",attributes:a,children:t}]}(O)}var tn=function(){},en=(_.measurePerformance&&y&&y.mark&&y.measure,function(n,t,e,r){var a,i,o,c=Object.keys(n),s=c.length,l=void 0!==r?function(n,t){return function(e,r,a,i){return n.call(t,e,r,a,i)}}(t,r):t;for(void 0===e?(a=1,o=n[c[0]]):(a=0,o=e);a<s;a++)o=l(o,n[i=c[a]],i,n);return o});function rn(n,t){var e=arguments.length>2&&void 0!==arguments[2]?arguments[2]:{},r=e.skipHooks,a=void 0!==r&&r,i=Object.keys(t).reduce((function(n,e){var r=t[e];return!!r.icon?n[r.iconName]=r.icon:n[e]=r,n}),{});"function"!=typeof T.hooks.addPack||a?T.styles[n]=c({},T.styles[n]||{},i):T.hooks.addPack(n,i),"fas"===n&&rn("fa",t)}var an=T.styles,on=T.shims,cn=function(){var n=function(n){return en(an,(function(t,e,r){return t[r]=en(e,n,{}),t}),{})};n((function(n,t,e){return t[3]&&(n[t[3]]=e),n})),n((function(n,t,e){var r=t[2];return n[e]=e,r.forEach((function(t){n[t]=e})),n}));var t="far"in an;en(on,(function(n,e){var r=e[0],a=e[1],i=e[2];return"far"!==a||t||(a="fas"),n[r]={prefix:a,iconName:i},n}),{})};cn();T.styles;function sn(n,t,e){if(n&&n[t]&&n[t][e])return{prefix:t,iconName:e,icon:n[t][e]}}function ln(n){var t=n.tag,e=n.attributes,r=void 0===e?{}:e,a=n.children,i=void 0===a?[]:a;return"string"==typeof n?K(n):"<".concat(t," ").concat(function(n){return Object.keys(n||{}).reduce((function(t,e){return t+"".concat(e,'="').concat(K(n[e]),'" ')}),"").trim()}(r),">").concat(i.map(ln).join(""),"</").concat(t,">")}var fn=function(n){var t={size:16,x:0,y:0,flipX:!1,flipY:!1,rotate:0};return n?n.toLowerCase().split(" ").reduce((function(n,t){var e=t.toLowerCase().split("-"),r=e[0],a=e.slice(1).join("-");if(r&&"h"===a)return n.flipX=!0,n;if(r&&"v"===a)return n.flipY=!0,n;if(a=parseFloat(a),isNaN(a))return n;switch(r){case"grow":n.size=n.size+a;break;case"shrink":n.size=n.size-a;break;case"left":n.x=n.x-a;break;case"right":n.x=n.x+a;break;case"up":n.y=n.y-a;break;case"down":n.y=n.y+a;break;case"rotate":n.rotate=n.rotate+a}return n}),t):t};function un(n){this.name="MissingIcon",this.message=n||"Icon unavailable",this.stack=(new Error).stack}un.prototype=Object.create(Error.prototype),un.prototype.constructor=un;var mn={fill:"currentColor"},dn={attributeType:"XML",repeatCount:"indefinite",dur:"2s"},pn={tag:"path",attributes:c({},mn,{d:"M156.5,447.7l-12.6,29.5c-18.7-9.5-35.9-21.2-51.5-34.9l22.7-22.7C127.6,430.5,141.5,440,156.5,447.7z M40.6,272H8.5 c1.4,21.2,5.4,41.7,11.7,61.1L50,321.2C45.1,305.5,41.8,289,40.6,272z M40.6,240c1.4-18.8,5.2-37,11.1-54.1l-29.5-12.6 C14.7,194.3,10,216.7,8.5,240H40.6z M64.3,156.5c7.8-14.9,17.2-28.8,28.1-41.5L69.7,92.3c-13.7,15.6-25.5,32.8-34.9,51.5 L64.3,156.5z M397,419.6c-13.9,12-29.4,22.3-46.1,30.4l11.9,29.8c20.7-9.9,39.8-22.6,56.9-37.6L397,419.6z M115,92.4 c13.9-12,29.4-22.3,46.1-30.4l-11.9-29.8c-20.7,9.9-39.8,22.6-56.8,37.6L115,92.4z M447.7,355.5c-7.8,14.9-17.2,28.8-28.1,41.5 l22.7,22.7c13.7-15.6,25.5-32.9,34.9-51.5L447.7,355.5z M471.4,272c-1.4,18.8-5.2,37-11.1,54.1l29.5,12.6 c7.5-21.1,12.2-43.5,13.6-66.8H471.4z M321.2,462c-15.7,5-32.2,8.2-49.2,9.4v32.1c21.2-1.4,41.7-5.4,61.1-11.7L321.2,462z M240,471.4c-18.8-1.4-37-5.2-54.1-11.1l-12.6,29.5c21.1,7.5,43.5,12.2,66.8,13.6V471.4z M462,190.8c5,15.7,8.2,32.2,9.4,49.2h32.1 c-1.4-21.2-5.4-41.7-11.7-61.1L462,190.8z M92.4,397c-12-13.9-22.3-29.4-30.4-46.1l-29.8,11.9c9.9,20.7,22.6,39.8,37.6,56.9 L92.4,397z M272,40.6c18.8,1.4,36.9,5.2,54.1,11.1l12.6-29.5C317.7,14.7,295.3,10,272,8.5V40.6z M190.8,50 c15.7-5,32.2-8.2,49.2-9.4V8.5c-21.2,1.4-41.7,5.4-61.1,11.7L190.8,50z M442.3,92.3L419.6,115c12,13.9,22.3,29.4,30.5,46.1 l29.8-11.9C470,128.5,457.3,109.4,442.3,92.3z M397,92.4l22.7-22.7c-15.6-13.7-32.8-25.5-51.5-34.9l-12.6,29.5 C370.4,72.1,384.4,81.5,397,92.4z"})},hn=c({},dn,{attributeName:"opacity"});c({},mn,{cx:"256",cy:"364",r:"28"}),c({},dn,{attributeName:"r",values:"28;14;28;28;14;28;"}),c({},hn,{values:"1;0;1;1;0;1;"}),c({},mn,{opacity:"1",d:"M263.7,312h-16c-6.6,0-12-5.4-12-12c0-71,77.4-63.9,77.4-107.8c0-20-17.8-40.2-57.4-40.2c-29.1,0-44.3,9.6-59.2,28.7 c-3.9,5-11.1,6-16.2,2.4l-13.1-9.2c-5.6-3.9-6.9-11.8-2.6-17.2c21.2-27.2,46.4-44.7,91.2-44.7c52.3,0,97.4,29.8,97.4,80.2 c0,67.6-77.4,63.5-77.4,107.8C275.7,306.6,270.3,312,263.7,312z"}),c({},hn,{values:"1;0;0;0;0;1;"}),c({},mn,{opacity:"0",d:"M232.5,134.5l7,168c0.3,6.4,5.6,11.5,12,11.5h9c6.4,0,11.7-5.1,12-11.5l7-168c0.3-6.8-5.2-12.5-12-12.5h-23 C237.7,122,232.2,127.7,232.5,134.5z"}),c({},hn,{values:"0;0;1;1;0;0;"}),T.styles;function gn(n){var t=n[0],e=n[1],r=s(n.slice(4),1)[0];return{found:!0,width:t,height:e,icon:Array.isArray(r)?{tag:"g",attributes:{class:"".concat(_.familyPrefix,"-").concat(x.GROUP)},children:[{tag:"path",attributes:{class:"".concat(_.familyPrefix,"-").concat(x.SECONDARY),fill:"currentColor",d:r[0]}},{tag:"path",attributes:{class:"".concat(_.familyPrefix,"-").concat(x.PRIMARY),fill:"currentColor",d:r[1]}}]}:{tag:"path",attributes:{fill:"currentColor",d:r}}}}T.styles;function yn(){var n="svg-inline--fa",t=_.familyPrefix,e=_.replacementClass,r='svg:not(:root).svg-inline--fa {\n  overflow: visible;\n}\n\n.svg-inline--fa {\n  display: inline-block;\n  font-size: inherit;\n  height: 1em;\n  overflow: visible;\n  vertical-align: -0.125em;\n}\n.svg-inline--fa.fa-lg {\n  vertical-align: -0.225em;\n}\n.svg-inline--fa.fa-w-1 {\n  width: 0.0625em;\n}\n.svg-inline--fa.fa-w-2 {\n  width: 0.125em;\n}\n.svg-inline--fa.fa-w-3 {\n  width: 0.1875em;\n}\n.svg-inline--fa.fa-w-4 {\n  width: 0.25em;\n}\n.svg-inline--fa.fa-w-5 {\n  width: 0.3125em;\n}\n.svg-inline--fa.fa-w-6 {\n  width: 0.375em;\n}\n.svg-inline--fa.fa-w-7 {\n  width: 0.4375em;\n}\n.svg-inline--fa.fa-w-8 {\n  width: 0.5em;\n}\n.svg-inline--fa.fa-w-9 {\n  width: 0.5625em;\n}\n.svg-inline--fa.fa-w-10 {\n  width: 0.625em;\n}\n.svg-inline--fa.fa-w-11 {\n  width: 0.6875em;\n}\n.svg-inline--fa.fa-w-12 {\n  width: 0.75em;\n}\n.svg-inline--fa.fa-w-13 {\n  width: 0.8125em;\n}\n.svg-inline--fa.fa-w-14 {\n  width: 0.875em;\n}\n.svg-inline--fa.fa-w-15 {\n  width: 0.9375em;\n}\n.svg-inline--fa.fa-w-16 {\n  width: 1em;\n}\n.svg-inline--fa.fa-w-17 {\n  width: 1.0625em;\n}\n.svg-inline--fa.fa-w-18 {\n  width: 1.125em;\n}\n.svg-inline--fa.fa-w-19 {\n  width: 1.1875em;\n}\n.svg-inline--fa.fa-w-20 {\n  width: 1.25em;\n}\n.svg-inline--fa.fa-pull-left {\n  margin-right: 0.3em;\n  width: auto;\n}\n.svg-inline--fa.fa-pull-right {\n  margin-left: 0.3em;\n  width: auto;\n}\n.svg-inline--fa.fa-border {\n  height: 1.5em;\n}\n.svg-inline--fa.fa-li {\n  width: 2em;\n}\n.svg-inline--fa.fa-fw {\n  width: 1.25em;\n}\n\n.fa-layers svg.svg-inline--fa {\n  bottom: 0;\n  left: 0;\n  margin: auto;\n  position: absolute;\n  right: 0;\n  top: 0;\n}\n\n.fa-layers {\n  display: inline-block;\n  height: 1em;\n  position: relative;\n  text-align: center;\n  vertical-align: -0.125em;\n  width: 1em;\n}\n.fa-layers svg.svg-inline--fa {\n  -webkit-transform-origin: center center;\n          transform-origin: center center;\n}\n\n.fa-layers-counter, .fa-layers-text {\n  display: inline-block;\n  position: absolute;\n  text-align: center;\n}\n\n.fa-layers-text {\n  left: 50%;\n  top: 50%;\n  -webkit-transform: translate(-50%, -50%);\n          transform: translate(-50%, -50%);\n  -webkit-transform-origin: center center;\n          transform-origin: center center;\n}\n\n.fa-layers-counter {\n  background-color: #ff253a;\n  border-radius: 1em;\n  -webkit-box-sizing: border-box;\n          box-sizing: border-box;\n  color: #fff;\n  height: 1.5em;\n  line-height: 1;\n  max-width: 5em;\n  min-width: 1.5em;\n  overflow: hidden;\n  padding: 0.25em;\n  right: 0;\n  text-overflow: ellipsis;\n  top: 0;\n  -webkit-transform: scale(0.25);\n          transform: scale(0.25);\n  -webkit-transform-origin: top right;\n          transform-origin: top right;\n}\n\n.fa-layers-bottom-right {\n  bottom: 0;\n  right: 0;\n  top: auto;\n  -webkit-transform: scale(0.25);\n          transform: scale(0.25);\n  -webkit-transform-origin: bottom right;\n          transform-origin: bottom right;\n}\n\n.fa-layers-bottom-left {\n  bottom: 0;\n  left: 0;\n  right: auto;\n  top: auto;\n  -webkit-transform: scale(0.25);\n          transform: scale(0.25);\n  -webkit-transform-origin: bottom left;\n          transform-origin: bottom left;\n}\n\n.fa-layers-top-right {\n  right: 0;\n  top: 0;\n  -webkit-transform: scale(0.25);\n          transform: scale(0.25);\n  -webkit-transform-origin: top right;\n          transform-origin: top right;\n}\n\n.fa-layers-top-left {\n  left: 0;\n  right: auto;\n  top: 0;\n  -webkit-transform: scale(0.25);\n          transform: scale(0.25);\n  -webkit-transform-origin: top left;\n          transform-origin: top left;\n}\n\n.fa-lg {\n  font-size: 1.3333333333em;\n  line-height: 0.75em;\n  vertical-align: -0.0667em;\n}\n\n.fa-xs {\n  font-size: 0.75em;\n}\n\n.fa-sm {\n  font-size: 0.875em;\n}\n\n.fa-1x {\n  font-size: 1em;\n}\n\n.fa-2x {\n  font-size: 2em;\n}\n\n.fa-3x {\n  font-size: 3em;\n}\n\n.fa-4x {\n  font-size: 4em;\n}\n\n.fa-5x {\n  font-size: 5em;\n}\n\n.fa-6x {\n  font-size: 6em;\n}\n\n.fa-7x {\n  font-size: 7em;\n}\n\n.fa-8x {\n  font-size: 8em;\n}\n\n.fa-9x {\n  font-size: 9em;\n}\n\n.fa-10x {\n  font-size: 10em;\n}\n\n.fa-fw {\n  text-align: center;\n  width: 1.25em;\n}\n\n.fa-ul {\n  list-style-type: none;\n  margin-left: 2.5em;\n  padding-left: 0;\n}\n.fa-ul > li {\n  position: relative;\n}\n\n.fa-li {\n  left: -2em;\n  position: absolute;\n  text-align: center;\n  width: 2em;\n  line-height: inherit;\n}\n\n.fa-border {\n  border: solid 0.08em #eee;\n  border-radius: 0.1em;\n  padding: 0.2em 0.25em 0.15em;\n}\n\n.fa-pull-left {\n  float: left;\n}\n\n.fa-pull-right {\n  float: right;\n}\n\n.fa.fa-pull-left,\n.fas.fa-pull-left,\n.far.fa-pull-left,\n.fal.fa-pull-left,\n.fab.fa-pull-left {\n  margin-right: 0.3em;\n}\n.fa.fa-pull-right,\n.fas.fa-pull-right,\n.far.fa-pull-right,\n.fal.fa-pull-right,\n.fab.fa-pull-right {\n  margin-left: 0.3em;\n}\n\n.fa-spin {\n  -webkit-animation: fa-spin 2s infinite linear;\n          animation: fa-spin 2s infinite linear;\n}\n\n.fa-pulse {\n  -webkit-animation: fa-spin 1s infinite steps(8);\n          animation: fa-spin 1s infinite steps(8);\n}\n\n@-webkit-keyframes fa-spin {\n  0% {\n    -webkit-transform: rotate(0deg);\n            transform: rotate(0deg);\n  }\n  100% {\n    -webkit-transform: rotate(360deg);\n            transform: rotate(360deg);\n  }\n}\n\n@keyframes fa-spin {\n  0% {\n    -webkit-transform: rotate(0deg);\n            transform: rotate(0deg);\n  }\n  100% {\n    -webkit-transform: rotate(360deg);\n            transform: rotate(360deg);\n  }\n}\n.fa-rotate-90 {\n  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=1)";\n  -webkit-transform: rotate(90deg);\n          transform: rotate(90deg);\n}\n\n.fa-rotate-180 {\n  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2)";\n  -webkit-transform: rotate(180deg);\n          transform: rotate(180deg);\n}\n\n.fa-rotate-270 {\n  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=3)";\n  -webkit-transform: rotate(270deg);\n          transform: rotate(270deg);\n}\n\n.fa-flip-horizontal {\n  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=0, mirror=1)";\n  -webkit-transform: scale(-1, 1);\n          transform: scale(-1, 1);\n}\n\n.fa-flip-vertical {\n  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1)";\n  -webkit-transform: scale(1, -1);\n          transform: scale(1, -1);\n}\n\n.fa-flip-both, .fa-flip-horizontal.fa-flip-vertical {\n  -ms-filter: "progid:DXImageTransform.Microsoft.BasicImage(rotation=2, mirror=1)";\n  -webkit-transform: scale(-1, -1);\n          transform: scale(-1, -1);\n}\n\n:root .fa-rotate-90,\n:root .fa-rotate-180,\n:root .fa-rotate-270,\n:root .fa-flip-horizontal,\n:root .fa-flip-vertical,\n:root .fa-flip-both {\n  -webkit-filter: none;\n          filter: none;\n}\n\n.fa-stack {\n  display: inline-block;\n  height: 2em;\n  position: relative;\n  width: 2.5em;\n}\n\n.fa-stack-1x,\n.fa-stack-2x {\n  bottom: 0;\n  left: 0;\n  margin: auto;\n  position: absolute;\n  right: 0;\n  top: 0;\n}\n\n.svg-inline--fa.fa-stack-1x {\n  height: 1em;\n  width: 1.25em;\n}\n.svg-inline--fa.fa-stack-2x {\n  height: 2em;\n  width: 2.5em;\n}\n\n.fa-inverse {\n  color: #fff;\n}\n\n.sr-only {\n  border: 0;\n  clip: rect(0, 0, 0, 0);\n  height: 1px;\n  margin: -1px;\n  overflow: hidden;\n  padding: 0;\n  position: absolute;\n  width: 1px;\n}\n\n.sr-only-focusable:active, .sr-only-focusable:focus {\n  clip: auto;\n  height: auto;\n  margin: 0;\n  overflow: visible;\n  position: static;\n  width: auto;\n}\n\n.svg-inline--fa .fa-primary {\n  fill: var(--fa-primary-color, currentColor);\n  opacity: 1;\n  opacity: var(--fa-primary-opacity, 1);\n}\n\n.svg-inline--fa .fa-secondary {\n  fill: var(--fa-secondary-color, currentColor);\n  opacity: 0.4;\n  opacity: var(--fa-secondary-opacity, 0.4);\n}\n\n.svg-inline--fa.fa-swap-opacity .fa-primary {\n  opacity: 0.4;\n  opacity: var(--fa-secondary-opacity, 0.4);\n}\n\n.svg-inline--fa.fa-swap-opacity .fa-secondary {\n  opacity: 1;\n  opacity: var(--fa-primary-opacity, 1);\n}\n\n.svg-inline--fa mask .fa-primary,\n.svg-inline--fa mask .fa-secondary {\n  fill: black;\n}\n\n.fad.fa-inverse {\n  color: #fff;\n}';if("fa"!==t||e!==n){var a=new RegExp("\\.".concat("fa","\\-"),"g"),i=new RegExp("\\--".concat("fa","\\-"),"g"),o=new RegExp("\\.".concat(n),"g");r=r.replace(a,".".concat(t,"-")).replace(i,"--".concat(t,"-")).replace(o,".".concat(e))}return r}function bn(){_.autoAddCss&&!On&&(q(yn()),On=!0)}function vn(n,t){return Object.defineProperty(n,"abstract",{get:t}),Object.defineProperty(n,"html",{get:function(){return n.abstract.map((function(n){return ln(n)}))}}),Object.defineProperty(n,"node",{get:function(){if(b){var t=g.createElement("div");return t.innerHTML=n.html,t.children}}}),n}function wn(n){var t=n.prefix,e=void 0===t?"fa":t,r=n.iconName;if(r)return sn(kn.definitions,e,r)||sn(T.styles,e,r)}var xn,kn=new(function(){function n(){!function(n,t){if(!(n instanceof t))throw new TypeError("Cannot call a class as a function")}(this,n),this.definitions={}}var t,e,r;return t=n,(e=[{key:"add",value:function(){for(var n=this,t=arguments.length,e=new Array(t),r=0;r<t;r++)e[r]=arguments[r];var a=e.reduce(this._pullDefinitions,{});Object.keys(a).forEach((function(t){n.definitions[t]=c({},n.definitions[t]||{},a[t]),rn(t,a[t]),cn()}))}},{key:"reset",value:function(){this.definitions={}}},{key:"_pullDefinitions",value:function(n,t){var e=t.prefix&&t.iconName&&t.icon?{0:t}:t;return Object.keys(e).map((function(t){var r=e[t],a=r.prefix,i=r.iconName,o=r.icon;n[a]||(n[a]={}),n[a][i]=o})),n}}])&&i(t.prototype,e),r&&i(t,r),n}()),On=!1,_n={transform:function(n){return fn(n)}},En=(xn=function(n){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{},e=t.transform,r=void 0===e?H:e,a=t.symbol,i=void 0!==a&&a,o=t.mask,s=void 0===o?null:o,l=t.maskId,f=void 0===l?null:l,u=t.title,m=void 0===u?null:u,d=t.titleId,p=void 0===d?null:d,h=t.classes,g=void 0===h?[]:h,y=t.attributes,b=void 0===y?{}:y,v=t.styles,w=void 0===v?{}:v;if(n){var x=n.prefix,k=n.iconName,O=n.icon;return vn(c({type:"icon"},n),(function(){return bn(),_.autoA11y&&(m?b["aria-labelledby"]="".concat(_.replacementClass,"-title-").concat(p||J()):(b["aria-hidden"]="true",b.focusable="false")),nn({icons:{main:gn(O),mask:s?gn(s.icon):{found:!1,width:null,height:null,icon:{}}},prefix:x,iconName:k,transform:c({},H,r),symbol:i,title:m,maskId:f,titleId:p,extra:{attributes:b,styles:w,classes:g}})}))}},function(n){var t=arguments.length>1&&void 0!==arguments[1]?arguments[1]:{},e=(n||{}).icon?n:wn(n||{}),r=t.mask;return r&&(r=(r||{}).icon?r:wn(r||{})),xn(e,c({},t,{mask:r}))})}).call(this,e("yLpj"),e("URgk").setImmediate)},"8oxB":function(n,t){var e,r,a=n.exports={};function i(){throw new Error("setTimeout has not been defined")}function o(){throw new Error("clearTimeout has not been defined")}function c(n){if(e===setTimeout)return setTimeout(n,0);if((e===i||!e)&&setTimeout)return e=setTimeout,setTimeout(n,0);try{return e(n,0)}catch(t){try{return e.call(null,n,0)}catch(t){return e.call(this,n,0)}}}!function(){try{e="function"==typeof setTimeout?setTimeout:i}catch(n){e=i}try{r="function"==typeof clearTimeout?clearTimeout:o}catch(n){r=o}}();var s,l=[],f=!1,u=-1;function m(){f&&s&&(f=!1,s.length?l=s.concat(l):u=-1,l.length&&d())}function d(){if(!f){var n=c(m);f=!0;for(var t=l.length;t;){for(s=l,l=[];++u<t;)s&&s[u].run();u=-1,t=l.length}s=null,f=!1,function(n){if(r===clearTimeout)return clearTimeout(n);if((r===o||!r)&&clearTimeout)return r=clearTimeout,clearTimeout(n);try{r(n)}catch(t){try{return r.call(null,n)}catch(t){return r.call(this,n)}}}(n)}}function p(n,t){this.fun=n,this.array=t}function h(){}a.nextTick=function(n){var t=new Array(arguments.length-1);if(arguments.length>1)for(var e=1;e<arguments.length;e++)t[e-1]=arguments[e];l.push(new p(n,t)),1!==l.length||f||c(d)},p.prototype.run=function(){this.fun.apply(null,this.array)},a.title="browser",a.browser=!0,a.env={},a.argv=[],a.version="",a.versions={},a.on=h,a.addListener=h,a.once=h,a.off=h,a.removeListener=h,a.removeAllListeners=h,a.emit=h,a.prependListener=h,a.prependOnceListener=h,a.listeners=function(n){return[]},a.binding=function(n){throw new Error("process.binding is not supported")},a.cwd=function(){return"/"},a.chdir=function(n){throw new Error("process.chdir is not supported")},a.umask=function(){return 0}},KaJK:function(n,t,e){var r=e("4jnk"),a=/"/g;n.exports=function(n,t,e,i){var o=String(r(n)),c="<"+t;return""!==e&&(c+=" "+e+'="'+String(i).replace(a,"&quot;")+'"'),c+">"+o+"</"+t+">"}},RXBc:function(n,t,e){"use strict";e.r(t);e("XCjP");var r=e("8tEE"),a=e("7O5W"),i=e("17x9"),o=e.n(i),c=e("q1tI"),s=e.n(c);function l(n){return(l="function"==typeof Symbol&&"symbol"==typeof Symbol.iterator?function(n){return typeof n}:function(n){return n&&"function"==typeof Symbol&&n.constructor===Symbol&&n!==Symbol.prototype?"symbol":typeof n})(n)}function f(n,t,e){return t in n?Object.defineProperty(n,t,{value:e,enumerable:!0,configurable:!0,writable:!0}):n[t]=e,n}function u(n,t){var e=Object.keys(n);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(n);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(n,t).enumerable}))),e.push.apply(e,r)}return e}function m(n){for(var t=1;t<arguments.length;t++){var e=null!=arguments[t]?arguments[t]:{};t%2?u(Object(e),!0).forEach((function(t){f(n,t,e[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(n,Object.getOwnPropertyDescriptors(e)):u(Object(e)).forEach((function(t){Object.defineProperty(n,t,Object.getOwnPropertyDescriptor(e,t))}))}return n}function d(n,t){if(null==n)return{};var e,r,a=function(n,t){if(null==n)return{};var e,r,a={},i=Object.keys(n);for(r=0;r<i.length;r++)e=i[r],t.indexOf(e)>=0||(a[e]=n[e]);return a}(n,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(n);for(r=0;r<i.length;r++)e=i[r],t.indexOf(e)>=0||Object.prototype.propertyIsEnumerable.call(n,e)&&(a[e]=n[e])}return a}function p(n){return function(n){if(Array.isArray(n)){for(var t=0,e=new Array(n.length);t<n.length;t++)e[t]=n[t];return e}}(n)||function(n){if(Symbol.iterator in Object(n)||"[object Arguments]"===Object.prototype.toString.call(n))return Array.from(n)}(n)||function(){throw new TypeError("Invalid attempt to spread non-iterable instance")}()}function h(n){return t=n,(t-=0)==t?n:(n=n.replace(/[\-_\s]+(.)?/g,(function(n,t){return t?t.toUpperCase():""}))).substr(0,1).toLowerCase()+n.substr(1);var t}function g(n){return n.split(";").map((function(n){return n.trim()})).filter((function(n){return n})).reduce((function(n,t){var e,r=t.indexOf(":"),a=h(t.slice(0,r)),i=t.slice(r+1).trim();return a.startsWith("webkit")?n[(e=a,e.charAt(0).toUpperCase()+e.slice(1))]=i:n[a]=i,n}),{})}var y=!1;try{y=!0}catch(A){}function b(n){return null===n?null:"object"===l(n)&&n.prefix&&n.iconName?n:Array.isArray(n)&&2===n.length?{prefix:n[0],iconName:n[1]}:"string"==typeof n?{prefix:"fas",iconName:n}:void 0}function v(n,t){return Array.isArray(t)&&t.length>0||!Array.isArray(t)&&t?f({},n,t):{}}function w(n){var t=n.forwardedRef,e=d(n,["forwardedRef"]),r=e.icon,i=e.mask,o=e.symbol,c=e.className,s=e.title,l=b(r),u=v("classes",[].concat(p(function(n){var t,e=n.spin,r=n.pulse,a=n.fixedWidth,i=n.inverse,o=n.border,c=n.listItem,s=n.flip,l=n.size,u=n.rotation,m=n.pull,d=(f(t={"fa-spin":e,"fa-pulse":r,"fa-fw":a,"fa-inverse":i,"fa-border":o,"fa-li":c,"fa-flip-horizontal":"horizontal"===s||"both"===s,"fa-flip-vertical":"vertical"===s||"both"===s},"fa-".concat(l),null!=l),f(t,"fa-rotate-".concat(u),null!=u&&0!==u),f(t,"fa-pull-".concat(m),null!=m),f(t,"fa-swap-opacity",n.swapOpacity),t);return Object.keys(d).map((function(n){return d[n]?n:null})).filter((function(n){return n}))}(e)),p(c.split(" ")))),h=v("transform","string"==typeof e.transform?a.c.transform(e.transform):e.transform),g=v("mask",b(i)),k=Object(a.b)(l,m({},u,{},h,{},g,{symbol:o,title:s}));if(!k)return function(){var n;!y&&console&&"function"==typeof console.error&&(n=console).error.apply(n,arguments)}("Could not find icon",l),null;var O=k.abstract,_={ref:t};return Object.keys(e).forEach((function(n){w.defaultProps.hasOwnProperty(n)||(_[n]=e[n])})),x(O[0],_)}w.displayName="FontAwesomeIcon",w.propTypes={border:o.a.bool,className:o.a.string,mask:o.a.oneOfType([o.a.object,o.a.array,o.a.string]),fixedWidth:o.a.bool,inverse:o.a.bool,flip:o.a.oneOf(["horizontal","vertical","both"]),icon:o.a.oneOfType([o.a.object,o.a.array,o.a.string]),listItem:o.a.bool,pull:o.a.oneOf(["right","left"]),pulse:o.a.bool,rotation:o.a.oneOf([0,90,180,270]),size:o.a.oneOf(["lg","xs","sm","1x","2x","3x","4x","5x","6x","7x","8x","9x","10x"]),spin:o.a.bool,symbol:o.a.oneOfType([o.a.bool,o.a.string]),title:o.a.string,transform:o.a.oneOfType([o.a.string,o.a.object]),swapOpacity:o.a.bool},w.defaultProps={border:!1,className:"",mask:null,fixedWidth:!1,inverse:!1,flip:null,icon:null,listItem:!1,pull:null,pulse:!1,rotation:null,size:null,spin:!1,symbol:!1,title:"",transform:null,swapOpacity:!1};var x=function n(t,e){var r=arguments.length>2&&void 0!==arguments[2]?arguments[2]:{};if("string"==typeof e)return e;var a=(e.children||[]).map((function(e){return n(t,e)})),i=Object.keys(e.attributes||{}).reduce((function(n,t){var r=e.attributes[t];switch(t){case"class":n.attrs.className=r,delete e.attributes.class;break;case"style":n.attrs.style=g(r);break;default:0===t.indexOf("aria-")||0===t.indexOf("data-")?n.attrs[t.toLowerCase()]=r:n.attrs[h(t)]=r}return n}),{attrs:{}}),o=r.style,c=void 0===o?{}:o,s=d(r,["style"]);return i.attrs.style=m({},i.attrs.style,{},c),t.apply(void 0,[e.tag,m({},i.attrs,{},s)].concat(p(a)))}.bind(null,s.a.createElement),k=e("9eSz"),O=e.n(k),_=e("vrFN"),E=e("Bl7J"),T=e("pyUO"),I=e.n(T);e("VAPu");a.a.autoAddCss=!1;var j=function(n){var t=n.data;return s.a.createElement(s.a.Fragment,null,s.a.createElement(_.a,null),s.a.createElement(E.a,null,s.a.createElement("div",{className:I.a.indexContainer},s.a.createElement("div",{className:I.a.indexRow},s.a.createElement("div",{className:I.a.textCol},s.a.createElement("h1",{className:I.a.heading},"Hello, I'm Tom."),s.a.createElement("p",{className:I.a.summary},"I am a data scientist and mathematician. I'm currently R&D Lead at"," ",s.a.createElement("a",{href:"https://faculty.ai"},"Faculty"))),s.a.createElement("div",{className:I.a.imgCol},s.a.createElement(O.a,{fixed:t.file.childImageSharp.fixed,alt:"tom",style:{borderRadius:8,width:200,height:200}}))),s.a.createElement("p",{className:I.a.text},"You've found my website. It has some information about me, and things I've worked on both professionally and in my spare time. I've also aspirationally set up a blog that I occasionally get around to writing things for. I'm interested in many things including, but not limited to Bayesian inference, machine learning, web development, and programming with Python."),s.a.createElement("span",{className:I.a.social},s.a.createElement("a",{href:"https://github.com/tcbegley"},s.a.createElement(w,{icon:r.a,size:"2x"})),s.a.createElement("a",{href:"https://linkedin.com/in/tcbegley"},s.a.createElement(w,{icon:r.b,size:"2x"})),s.a.createElement("a",{href:"https://twitter.com/tcbegley"},s.a.createElement(w,{icon:r.c,size:"2x"}))))))};j.propTypes={data:o.a.object.isRequired};t.default=j},URgk:function(n,t,e){(function(n){var r=void 0!==n&&n||"undefined"!=typeof self&&self||window,a=Function.prototype.apply;function i(n,t){this._id=n,this._clearFn=t}t.setTimeout=function(){return new i(a.call(setTimeout,r,arguments),clearTimeout)},t.setInterval=function(){return new i(a.call(setInterval,r,arguments),clearInterval)},t.clearTimeout=t.clearInterval=function(n){n&&n.close()},i.prototype.unref=i.prototype.ref=function(){},i.prototype.close=function(){this._clearFn.call(r,this._id)},t.enroll=function(n,t){clearTimeout(n._idleTimeoutId),n._idleTimeout=t},t.unenroll=function(n){clearTimeout(n._idleTimeoutId),n._idleTimeout=-1},t._unrefActive=t.active=function(n){clearTimeout(n._idleTimeoutId);var t=n._idleTimeout;t>=0&&(n._idleTimeoutId=setTimeout((function(){n._onTimeout&&n._onTimeout()}),t))},e("YBdB"),t.setImmediate="undefined"!=typeof self&&self.setImmediate||void 0!==n&&n.setImmediate||this&&this.setImmediate,t.clearImmediate="undefined"!=typeof self&&self.clearImmediate||void 0!==n&&n.clearImmediate||this&&this.clearImmediate}).call(this,e("yLpj"))},VAPu:function(n,t,e){},Wvaq:function(n,t,e){var r=e("JhOX");n.exports=function(n){return r((function(){var t=""[n]('"');return t!==t.toLowerCase()||t.split('"').length>3}))}},XCjP:function(n,t,e){"use strict";var r=e("ZS3K"),a=e("KaJK");r({target:"String",proto:!0,forced:e("Wvaq")("fixed")},{fixed:function(){return a(this,"tt","","")}})},YBdB:function(n,t,e){(function(n,t){!function(n,e){"use strict";if(!n.setImmediate){var r,a,i,o,c,s=1,l={},f=!1,u=n.document,m=Object.getPrototypeOf&&Object.getPrototypeOf(n);m=m&&m.setTimeout?m:n,"[object process]"==={}.toString.call(n.process)?r=function(n){t.nextTick((function(){p(n)}))}:!function(){if(n.postMessage&&!n.importScripts){var t=!0,e=n.onmessage;return n.onmessage=function(){t=!1},n.postMessage("","*"),n.onmessage=e,t}}()?n.MessageChannel?((i=new MessageChannel).port1.onmessage=function(n){p(n.data)},r=function(n){i.port2.postMessage(n)}):u&&"onreadystatechange"in u.createElement("script")?(a=u.documentElement,r=function(n){var t=u.createElement("script");t.onreadystatechange=function(){p(n),t.onreadystatechange=null,a.removeChild(t),t=null},a.appendChild(t)}):r=function(n){setTimeout(p,0,n)}:(o="setImmediate$"+Math.random()+"$",c=function(t){t.source===n&&"string"==typeof t.data&&0===t.data.indexOf(o)&&p(+t.data.slice(o.length))},n.addEventListener?n.addEventListener("message",c,!1):n.attachEvent("onmessage",c),r=function(t){n.postMessage(o+t,"*")}),m.setImmediate=function(n){"function"!=typeof n&&(n=new Function(""+n));for(var t=new Array(arguments.length-1),e=0;e<t.length;e++)t[e]=arguments[e+1];var a={callback:n,args:t};return l[s]=a,r(s),s++},m.clearImmediate=d}function d(n){delete l[n]}function p(n){if(f)setTimeout(p,0,n);else{var t=l[n];if(t){f=!0;try{!function(n){var t=n.callback,e=n.args;switch(e.length){case 0:t();break;case 1:t(e[0]);break;case 2:t(e[0],e[1]);break;case 3:t(e[0],e[1],e[2]);break;default:t.apply(void 0,e)}}(t)}finally{d(n),f=!1}}}}}("undefined"==typeof self?void 0===n?this:n:self)}).call(this,e("yLpj"),e("8oxB"))},pyUO:function(n,t,e){n.exports={indexContainer:"index-module--indexContainer--1TkSt",imgCol:"index-module--imgCol--3Ue6P",textCol:"index-module--textCol--2rxZo",indexRow:"index-module--indexRow--1x6cV",heading:"index-module--heading--3-2o-",summary:"index-module--summary--HxYhH",text:"index-module--text--2GCOn",social:"index-module--social--16L-W"}}}]);
//# sourceMappingURL=component---src-pages-index-js-b275f932f1d7f5a8e7f6.js.map