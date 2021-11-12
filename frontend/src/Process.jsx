import React  from "react";
import original from "../src/Image/original.png";
import edit from "../src/Image/edit.png";
import ability from "../src/Image/ability.png";
import ai from "../src/Image/aipowred.png"

const Process =()=>{
    return(
        <>

<section className="food-menu">
<h1 className="processabout">ABOUT</h1>
<div class="flex-container4">

  <div>
<img className="processimg" src={original} ></img>
<p className="processp" >ORIGINAL CONTENT</p>
  </div>
 
  <div>
    <img className="processimg" src={edit} ></img>
    <p className="processp">EASY MODIFY</p>
  </div>
  <div>
    <img className="processimg" src={ability} ></img>
    <p className="processp">AGILITY PROCESS</p>
  </div> 
  <div>
    <img className="processimg" src={ai} ></img>
  <p className="processp">AI POWERED</p>
  </div> 
</div>


  </section>

        </>
    )
}

export default Process;